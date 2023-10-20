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

#include "open_spiel/algorithms/ziggy_torch/model.h"

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

namespace open_spiel {
namespace algorithms {
namespace torch_az {

std::istream& operator>>(std::istream& stream, ModelConfig& config) {
  int channels;
  int height;
  int width;

  stream >> channels >> height >> width >>
      config.nn_depth >> config.nn_width >> config.learning_rate >>
      config.weight_decay >> config.nn_model;

  config.observation_tensor_shape = {channels, height, width};

  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ModelConfig& config) {
  stream << config.observation_tensor_shape[0] << " "
         << config.observation_tensor_shape[1] << " "
         << config.observation_tensor_shape[2] << " "
         << config.nn_depth << " "
         << config.nn_width << " " << config.learning_rate << " "
         << config.weight_decay << " " << config.nn_model;
  return stream;
}

ResInputBlockImpl::ResInputBlockImpl(const ResInputBlockConfig& config)
    : conv_(torch::nn::Conv2dOptions(
                /*input_channels=*/config.input_channels,
                /*output_channels=*/config.filters,
                /*kernel_size=*/config.kernel_size)
                .stride(1)
                .padding(config.padding)
                .dilation(1)
                .groups(1)
                .bias(true)
                //.bias(false)  // bias unnec when followed by batchnorm
                .padding_mode(torch::kZeros)),
      batch_norm_(torch::nn::BatchNorm2dOptions(
                      /*num_features=*/config.filters)
                      .eps(0.001)      // Make it the same as TF.
                      .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                      .affine(true)
                      .track_running_stats(true)) {
  channels_ = config.input_channels;
  height_ = config.input_height;
  width_ = config.input_width;

  register_module("input_conv", conv_);
  register_module("input_batch_norm", batch_norm_);
}

torch::Tensor ResInputBlockImpl::forward(torch::Tensor x) {
  torch::Tensor output = x.view({-1, channels_, height_, width_});
  output = torch::relu(batch_norm_(conv_(output)));

  return output;
}

ResTorsoBlockImpl::ResTorsoBlockImpl(const ResTorsoBlockConfig& config,
                                     int layer)
    : conv1_(torch::nn::Conv2dOptions(
                 /*input_channels=*/config.input_channels,
                 /*output_channels=*/config.filters,
                 /*kernel_size=*/config.kernel_size)
                 .stride(1)
                 .padding(config.padding)
                 .dilation(1)
                 .groups(1)
                 .bias(true)
                 //.bias(false)  // bias unnec when followed by batchnorm
                 .padding_mode(torch::kZeros)),
      conv2_(torch::nn::Conv2dOptions(
                 /*input_channels=*/config.filters,
                 /*output_channels=*/config.filters,
                 /*kernel_size=*/config.kernel_size)
                 .stride(1)
                 .padding(config.padding)
                 .dilation(1)
                 .groups(1)
                 .bias(true)
                 //.bias(false)  // bias unnec when followed by batchnorm
                 .padding_mode(torch::kZeros)),
      batch_norm1_(torch::nn::BatchNorm2dOptions(
                       /*num_features=*/config.filters)
                       .eps(0.001)      // Make it the same as TF.
                       .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                       .affine(true)
                       .track_running_stats(true)),
      batch_norm2_(torch::nn::BatchNorm2dOptions(
                       /*num_features=*/config.filters)
                       .eps(0.001)      // Make it the same as TF.
                       .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                       .affine(true)
                       .track_running_stats(true)) {
  register_module("res_" + std::to_string(layer) + "_conv_1", conv1_);
  register_module("res_" + std::to_string(layer) + "_conv_2", conv2_);
  register_module("res_" + std::to_string(layer) + "_batch_norm_1",
                  batch_norm1_);
  register_module("res_" + std::to_string(layer) + "_batch_norm_2",
                  batch_norm2_);
}

torch::Tensor ResTorsoBlockImpl::forward(torch::Tensor x) {
  torch::Tensor residual = x;

  torch::Tensor output = torch::relu(batch_norm1_(conv1_(x)));
  output = batch_norm2_(conv2_(output));
  output += residual;
  output = torch::relu(output);

  return output;
}

ResOutputBlockImpl::ResOutputBlockImpl(const ResOutputBlockConfig& config)
    : value_conv_(torch::nn::Conv2dOptions(
                      /*input_channels=*/config.input_channels,
                      /*output_channels=*/config.value_filters,
                      /*kernel_size=*/config.kernel_size)
                      .stride(1)
                      .padding(config.padding)
                      .dilation(1)
                      .groups(1)
                      .bias(true)
                      //.bias(false)  // bias unnec when followed by batchnorm
                      .padding_mode(torch::kZeros)),
      value_batch_norm_(
          torch::nn::BatchNorm2dOptions(
              /*num_features=*/config.value_filters)
              .eps(0.001)      // Make it the same as TF.
              .momentum(0.01)  // Torch momentum = 1 - TF momentum.
              .affine(true)
              .track_running_stats(true)),
      value_linear1_(torch::nn::LinearOptions(
                         /*in_features=*/config.value_linear_in_features,
                         /*out_features=*/config.value_linear_out_features)
                         .bias(true)),
      value_linear2_(torch::nn::LinearOptions(
                         /*in_features=*/config.value_linear_out_features,
                         /*out_features=*/1)
                         .bias(true)),
      value_observation_size_(config.value_observation_size) {
  register_module("value_conv", value_conv_);
  register_module("value_batch_norm", value_batch_norm_);
  register_module("value_linear_1", value_linear1_);
  register_module("value_linear_2", value_linear2_);
}

std::vector<torch::Tensor> ResOutputBlockImpl::forward(torch::Tensor x) {
  torch::Tensor value_output = torch::relu(value_batch_norm_(value_conv_(x)));
  value_output = value_output.view({-1, value_observation_size_});
  value_output = torch::relu(value_linear1_(value_output));
  value_output = torch::tanh(value_linear2_(value_output));
  return {value_output};
}

MLPTorsoBlockImpl::MLPTorsoBlockImpl(const int in_features,
                                     const int out_features)
    : linear_(torch::nn::LinearOptions(
                         /*in_features=*/in_features,
                         /*out_features=*/out_features)
                         .bias(true)) {
  register_module("linear", linear_);
}

torch::Tensor MLPTorsoBlockImpl::forward(torch::Tensor x) {
  return torch::leaky_relu(linear_(x));
}

MLPOutputBlockImpl::MLPOutputBlockImpl(const int nn_width)
    : value_linear1_(torch::nn::LinearOptions(
                         /*in_features=*/nn_width,
                         /*out_features=*/nn_width)
                         .bias(true)),
      value_linear2_(torch::nn::LinearOptions(
                         /*in_features=*/nn_width,
                         /*out_features=*/1)
                         .bias(true)) {
  register_module("value_linear_1", value_linear1_);
  register_module("value_linear_2", value_linear2_);
}

std::vector<torch::Tensor> MLPOutputBlockImpl::forward(torch::Tensor x) {
  torch::Tensor value_output = torch::leaky_relu(value_linear1_(x));
  value_output = torch::tanh(value_linear2_(value_output));
  return {value_output};
}

ModelImpl::ModelImpl(const ModelConfig& config, const std::string& device)
    : device_(device),
      num_torso_blocks_(config.nn_depth),
      weight_decay_(config.weight_decay) {

  // It may be this improves performance on other devices too, but it has
  // only been tested w cpu.
  if (device.find("cpu") != std::string::npos) {
    // libtorch-1.12 threading causes too much overhead on cpu,
    // Setting threads=1 speeds up performance significantly.
    // See also: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
    torch::set_num_threads(1);
  }
  std::cerr << "Torch numthreads=" << torch::get_num_threads() << std::endl;
  std::cerr << "Model learning rate=" << config.learning_rate << std::endl;
  std::cerr << "Model weight decay=" << config.weight_decay << std::endl;

  // Save config.nn_model to class
  nn_model_ = config.nn_model;

  int input_size = 1;
  for (const auto& num : config.observation_tensor_shape) {
    if (num > 0) {
      input_size *= num;
    }
  }
  // Decide if resnet or MLP
  if (config.nn_model == "resnet") {
    int channels = config.observation_tensor_shape[0];
    int height = config.observation_tensor_shape[1];
    int width = config.observation_tensor_shape[2];

    ResInputBlockConfig input_config = {/*input_channels=*/channels,
                                        /*input_height=*/height,
                                        /*input_width=*/width,
                                        /*filters=*/config.nn_width,
                                        /*kernel_size=*/3,
                                        /*padding=*/1};

    ResTorsoBlockConfig residual_config = {/*input_channels=*/config.nn_width,
                                           /*filters=*/config.nn_width,
                                           /*kernel_size=*/3,
                                           /*padding=*/1};

    ResOutputBlockConfig output_config = {
        /*input_channels=*/config.nn_width,
        /*value_filters=*/1,
        /*kernel_size=*/1,
        /*padding=*/0,
        /*value_linear_in_features=*/1 * width * height,
        /*value_linear_out_features=*/config.nn_width,
        /*value_observation_size=*/1 * width * height};

    layers_->push_back(ResInputBlock(input_config));
    for (int i = 0; i < num_torso_blocks_; i++) {
      layers_->push_back(ResTorsoBlock(residual_config, i));
    }
    layers_->push_back(ResOutputBlock(output_config));

    register_module("layers", layers_);

  } else if (config.nn_model == "mlp") {
    for (int i = 0; i < num_torso_blocks_; i++) {
      layers_->push_back(
          MLPTorsoBlock((i == 0 ? input_size : config.nn_width), config.nn_width));
    }
    layers_->push_back(
        MLPOutputBlock(config.nn_width));

    register_module("layers", layers_);
  } else {
    throw std::runtime_error("Unknown nn_model: " + config.nn_model);
  }
}

std::vector<torch::Tensor> ModelImpl::forward(torch::Tensor x) {
  std::vector<torch::Tensor> output = this->forward_(x);
  return {output[0]};
}

std::vector<torch::Tensor> ModelImpl::losses(torch::Tensor inputs,
                                             torch::Tensor value_targets) {
  std::vector<torch::Tensor> output = this->forward_(inputs);
  torch::Tensor value_predictions = output[0];

  // Value loss (mean-squared error).
  torch::nn::MSELoss mse_loss;
  torch::Tensor value_loss = mse_loss(value_predictions, value_targets);

  // L2 regularization loss (weights only).
  torch::Tensor l2_regularization_loss = torch::full(
      {1, 1}, 0, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  for (auto& named_parameter : this->named_parameters()) {
    // named_parameter is essentially a key-value pair:
    //   {key, value} == {std::string name, torch::Tensor parameter}
    std::string parameter_name = named_parameter.key();

    // Do not include bias' in the loss.
    if (parameter_name.find("bias") != std::string::npos) {
      continue;
    }

    // Copy TensorFlow's l2_loss function.
    // https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    l2_regularization_loss +=
        weight_decay_ * torch::sum(torch::square(named_parameter.value())) / 2;
  }

  return {value_loss, l2_regularization_loss};
}

std::vector<torch::Tensor> ModelImpl::forward_(torch::Tensor x) {
  std::vector<torch::Tensor> output;
  if (this->nn_model_ == "resnet") {
    for (int i = 0; i < num_torso_blocks_ + 2; i++) {
      if (i == 0) {
        x = layers_[i]->as<ResInputBlock>()->forward(x);
      } else if (i >= num_torso_blocks_ + 1) {
        output = layers_[i]->as<ResOutputBlock>()->forward(x);
      } else {
        x = layers_[i]->as<ResTorsoBlock>()->forward(x);
      }
    }
  } else if (this->nn_model_ == "mlp") {
    for (int i = 0; i < num_torso_blocks_; i++) {
        x = layers_[i]->as<MLPTorsoBlock>()->forward(x);
    }
    output = layers_[num_torso_blocks_]->as<MLPOutputBlockImpl>()->forward(x);
  } else {
    throw std::runtime_error("Unknown nn_model: " + this->nn_model_);
  }
  return output;
}

void ModelImpl::print() const {
  std::cerr << "Model parameters: " << std::endl;
  for (auto& named_parameter : this->named_parameters()) {
    std::cerr << named_parameter.key() << ": " << named_parameter.value() << std::endl;
  }
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
