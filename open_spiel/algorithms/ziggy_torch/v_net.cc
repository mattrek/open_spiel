// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/ziggy_torch/v_net.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <fstream>  // For ifstream/ofstream.
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/run_python.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

// Saves a struct that holds initialization data for the model to a file.
//
// The TensorFlow version creates a TensorFlow graph definition when
// CreateGraphDef is called. To avoid having to change this, allow calls to
// CreateGraphDef, however now it simply saves a struct to a file which can
// then be loaded and used to initialize a model.
bool SaveModelConfig(const std::string& path, const std::string& filename,
                     const ModelConfig& net_config) {
  std::ofstream file;
  file.open(absl::StrCat(path, "/", filename));

  if (!file) {
    return false;
  } else {
    file << net_config;
  }
  file.close();

  return true;
}

// Loads a struct that holds initialization data for the model from a file.
//
// The TensorFlow version creates a TensorFlow graph definition when
// CreateGraphDef is called. To avoid having to change this, allow calls to
// CreateGraphDef, however now it simply saves a struct to a file which can
// then be loaded and used to initialize a model.
ModelConfig LoadModelConfig(const std::string& path,
                            const std::string& filename) {
  std::ifstream file;
  file.open(absl::StrCat(path, "/", filename));
  ModelConfig net_config;

  file >> net_config;
  file.close();

  return net_config;
}

// Modifies a given device string to one that can be accepted by the
// Torch library.
//
// The Torch library accepts 'cpu', 'cpu:0', 'cuda:0', 'cuda:1',
// 'cuda:2', 'cuda:3'..., but complains when there's a slash in front
// of the device name.
//
// Currently, this function only disregards a slash if it exists at the
// beginning of the device string, more functionality can be added if
// needed.
std::string TorchDeviceName(const std::string& device) {
  if (device[0] == '/') {
    return device.substr(1);
  }
  return device;
}

bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& filename,
                    std::string nn_model, int nn_width, int nn_depth,
                    bool verbose) {
  ModelConfig net_config = {
      /*observation_tensor_shape=*/game.ObservationTensorShape(),
     // /*number_of_actions=*/game.NumDistinctActions(),
      /*nn_depth=*/nn_depth,
      /*nn_width=*/nn_width,
      /*learning_rate=*/learning_rate,
      /*weight_decay=*/weight_decay,
      /*nn_model=*/nn_model};

  return SaveModelConfig(path, filename, net_config);
}

VPNetModel::VPNetModel(const Game &game, const std::string &path,
                       const std::string &file_name, const std::string &device)
    : device_(device), path_(path),
      flat_input_size_(game.ObservationTensorSize()),
      //num_actions_(game.NumDistinctActions()),
      model_config_(LoadModelConfig(path, file_name)),
      torch_device_(TorchDeviceName(device)),
      model_(model_config_, TorchDeviceName(device)),
      model_optimizer_(model_->parameters(),
                       torch::optim::AdamOptions(model_config_.learning_rate)) {
  // Some assumptions that we can remove eventually. The value net returns
  // a single value in terms of player 0 and the game is assumed to be zero-sum,
  // so player 1 can just be -value.
  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);

  // Put this model on the specified device.
  model_->to(torch_device_);
}

std::string VPNetModel::SaveCheckpoint(int step) {
  std::string full_path = absl::StrCat(path_, "/checkpoint-", step);

  torch::save(model_, absl::StrCat(full_path, ".pt"));
  torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));

  return full_path;
}

void VPNetModel::LoadCheckpoint(int step) {
  // Load checkpoint from the path given at its initialization.
  LoadCheckpoint(absl::StrCat(path_, "/checkpoint-", step));
}

void VPNetModel::LoadCheckpoint(const std::string& path) {
  torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
  torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"),
              torch_device_);
}

std::vector<VPNetModel::InferenceOutputs> VPNetModel::Inference(
    const std::vector<InferenceInputs>& inputs) {
  int inference_batch_size = inputs.size();

  // Torch tensors by default use a dense, row-aligned memory layout.
  //   - Their default data type is a 32-bit float
  //   - Use the byte data type for boolean  (mattrek: obsolete, torch has bool)

  torch::Tensor torch_inf_inputs =
      torch::empty({inference_batch_size, flat_input_size_}, torch_device_);
  for (int batch = 0; batch < inference_batch_size; ++batch) {
    // Copy the observation(s) to a Torch tensor.
    for (int i = 0; i < inputs[batch].observations.size(); ++i) {
      torch_inf_inputs[batch][i] = inputs[batch].observations[i];
    }
  }

  // Run the inference.
  model_->eval();
  std::vector<torch::Tensor> torch_outputs = model_(torch_inf_inputs);
  torch::Tensor value_batch = torch_outputs[0];

  // Copy the Torch tensor output to the appropriate structure.
  std::vector<InferenceOutputs> output;
  output.reserve(inference_batch_size);
  for (int batch = 0; batch < inference_batch_size; ++batch) {
    double value = value_batch[batch].item<double>();
    output.push_back({value});
  }

  return output;
}

VPNetModel::LossInfo VPNetModel::Learn(const std::vector<TrainInputs>& inputs) {
  int training_batch_size = inputs.size();

  // Torch tensors by default use a dense, row-aligned memory layout.
  //   - Their default data type is a 32-bit float
  //   - Use the byte data type for boolean  (mattrek: obsolete, torch has bool)

  torch::Tensor torch_train_inputs =
      torch::empty({training_batch_size, flat_input_size_}, torch_device_);
  torch::Tensor torch_value_targets =
      torch::empty({training_batch_size, 1}, torch_device_);

  for (int batch = 0; batch < training_batch_size; ++batch) {
    // Copy the observation(s) to a Torch tensor.
    for (int i = 0; i < inputs[batch].observations.size(); ++i) {
      torch_train_inputs[batch][i] = inputs[batch].observations[i];
    }
    // Copy the value target(s) to a Torch tensor.
    torch_value_targets[batch][0] = inputs[batch].value;
  }

  // Run a training step and get the losses.
  model_->train();
  model_->zero_grad();

  std::vector<torch::Tensor> torch_outputs =
      model_->losses(torch_train_inputs, torch_value_targets);

  torch::Tensor total_loss = torch_outputs[0] + torch_outputs[1];
  total_loss.backward();
  model_optimizer_.step();

  return LossInfo(torch_outputs[0].item<float>(),
                  torch_outputs[1].item<float>());
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
