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
#ifdef TF_FE_NO_TF_DEP
    OVTFGraphIterator(shared_ptr<OVTFDecoder> head_ptr, int num_nodes) : m_head_ptr(head_ptr), m_current_ptr(m_head_ptr), m_num_nodes(num_nodes) {
    }
#else
    OVTFGraphIterator(const ::tensorflow::GraphDef *graph_def) : m_graph_def(graph_def) {
        m_nodes.resize(m_graph_def->node_size());
        for (size_t i = 0; i < m_nodes.size(); ++i)
            m_nodes[i] = &m_graph_def->node(i);
    }
#endif

    /// Set iterator to the start position
    void reset() override {
#ifdef TF_FE_NO_TF_DEP
      m_current_ptr = m_head_ptr;
#else
      node_index = 0;
#endif
    }

    size_t size() const override {
#ifdef TF_FE_NO_TF_DEP
      return m_num_nodes;
#else
      return m_nodes.size();
#endif
    }

    /// Moves to the next node in the graph
    void next() override {
#ifdef TF_FE_NO_TF_DEP
      if (m_current_ptr != nullptr)
        m_current_ptr = m_current_ptr->get_next();
#else
      node_index++;
#endif
    }

    bool is_end() const override {
#ifdef TF_FE_NO_TF_DEP
      if (m_current_ptr == nullptr)
        return true;
      else
        return ((m_current_ptr->get_next() == nullptr) ? true:false);
#else
      return node_index >= m_nodes.size();
#endif
    }

    /// Return NodeContext for the current node that iterator points to
    std::shared_ptr<ov::frontend::DecoderBase> get_decoder() const override {
#ifdef TF_FE_NO_TF_DEP
        return m_current_ptr;
#else
        return std::make_shared<OVTFDecoder>(m_nodes[node_index]);
#endif
    }

private:
#ifdef TF_FE_NO_TF_DEP
    shared_ptr<OVTFDecoder> m_head_ptr;
    shared_ptr<OVTFDecoder> m_current_ptr;
    int m_num_nodes;
#else
    std::vector<const ::tensorflow::NodeDef*> m_nodes;
    size_t node_index = 0;
    const ::tensorflow::GraphDef *m_graph_def;
#endif
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
