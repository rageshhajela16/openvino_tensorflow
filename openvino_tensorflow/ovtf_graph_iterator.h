/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"
#include <tensorflow_frontend/graph_iterator.hpp>
#include <tensorflow_frontend/decoder.hpp>
#include "openvino_tensorflow/ovtf_decoder.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// A Inference Engine executable object produced by compiling an nGraph
// function.
class OVTFGraphIterator : public ov::frontend::GraphIterator {
public:
    OVTFGraphIterator(shared_ptr<OVTFDecoder> head_ptr, int num_nodes) : m_head_ptr(head_ptr), m_current_ptr(m_head_ptr), m_num_nodes(num_nodes) {
    }

    /// Set iterator to the start position
    void reset() override {
      m_current_ptr = m_head_ptr;
    }

    size_t size() const override {
      return m_num_nodes;
    }

    /// Moves to the next node in the graph
    void next() override {
      if (m_current_ptr != nullptr)
        m_current_ptr = m_current_ptr->get_next();
    }

    bool is_end() const override {
      if (m_current_ptr == nullptr)
        return true;
      else
        return ((m_current_ptr->get_next() == nullptr) ? true:false);
    }

    /// Return NodeContext for the current node that iterator points to
    std::shared_ptr<ov::frontend::DecoderBase> get_decoder() const override {
        return m_current_ptr;
    }

private:
    shared_ptr<OVTFDecoder> m_head_ptr;
    shared_ptr<OVTFDecoder> m_current_ptr;
    int m_num_nodes;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
