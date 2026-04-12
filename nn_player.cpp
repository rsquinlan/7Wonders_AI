#include "nn_player.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>

NNPlayer::NNPlayer(const std::string& model_path, int num_players)
    : num_players_(num_players),
      env_(ORT_LOGGING_LEVEL_WARNING, "NNPlayer"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{}),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // The model has one input ("state_embedding") and two outputs ("policy", "value").
    // These string literals are stable for the lifetime of this object.
    input_names_  = {"state_embedding"};
    output_names_ = {"policy", "value"};
}

void NNPlayer::infer(const std::vector<float>& state_embedding,
                     std::vector<float>&        out_policy,
                     float&                     out_value)
{
    const int64_t input_size = static_cast<int64_t>(state_embedding.size());
    std::array<int64_t, 2> input_shape{1, input_size};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        const_cast<float*>(state_embedding.data()),
        state_embedding.size(),
        input_shape.data(), input_shape.size()
    );

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),  &input_tensor, 1,
        output_names_.data(), output_names_.size()
    );

    // Policy: shape [1, 75]
    const float* policy_ptr = outputs[0].GetTensorData<float>();
    out_policy.assign(policy_ptr, policy_ptr + 75);

    // Value: shape [1, 1] — logits, apply sigmoid to get probability
    float logit = *outputs[1].GetTensorData<float>();
    out_value = 1.0f / (1.0f + std::exp(-logit));
}

void NNPlayer::inferBatch(const std::vector<std::vector<float>>& embeddings,
                           std::vector<std::vector<float>>&        out_policies,
                           std::vector<float>&                     out_values)
{
    if (embeddings.empty()) return;

    const int64_t batch      = static_cast<int64_t>(embeddings.size());
    const int64_t input_size = static_cast<int64_t>(embeddings[0].size());

    // Flatten to a single contiguous buffer [batch, input_size]
    std::vector<float> flat;
    flat.reserve(batch * input_size);
    for (const auto& e : embeddings)
        flat.insert(flat.end(), e.begin(), e.end());

    std::array<int64_t, 2> shape{batch, input_size};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, flat.data(), flat.size(), shape.data(), shape.size());

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());

    const float* policy_ptr = outputs[0].GetTensorData<float>();  // [batch, 75]
    const float* value_ptr  = outputs[1].GetTensorData<float>();  // [batch, 1]

    out_policies.resize(batch);
    out_values.resize(batch);
    for (int64_t b = 0; b < batch; ++b) {
        out_policies[b].assign(policy_ptr + b * 75, policy_ptr + (b + 1) * 75);
        // Value is logits, apply sigmoid to get probability
        float logit = value_ptr[b];
        out_values[b] = 1.0f / (1.0f + std::exp(-logit));
    }
}

DMAG::Card NNPlayer::getBestMoveNN(const std::shared_ptr<DMAG::Game>& game, int playerIndex)
{
    auto playable = game->player_list[playerIndex]->GetPlayableCards();
    if (playable.empty()) {
        const auto& hand = game->player_list[playerIndex]->GetHandCards();
        if (!hand.empty()) return hand.front();
        // Should never happen in a valid game state
        throw std::runtime_error("NNPlayer: no cards available for player " + std::to_string(playerIndex));
    }

    std::vector<float> policy;
    float value;
    infer(encodeGameState(*game, playerIndex), policy, value);

    // Pick the playable card with the highest policy probability
    DMAG::Card best_card = playable.front();
    float      best_prob = -1.0f;

    for (const auto& card : playable) {
        int idx = card.GetId() - 1;  // policy vector is 0-indexed by card ID
        if (idx >= 0 && idx < static_cast<int>(policy.size())) {
            if (policy[idx] > best_prob) {
                best_prob = policy[idx];
                best_card = card;
            }
        }
    }

    return best_card;
}
