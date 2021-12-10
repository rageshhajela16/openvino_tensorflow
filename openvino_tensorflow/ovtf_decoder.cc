
#include <openvino_tensorflow/ovtf_decoder.h>
#include <ngraph/ngraph.hpp>
//#include <ngraph/variant.hpp>
//#include <ngraph/node_context.hpp>
#include <openvino/core/type/element_type.hpp>
#include <string>
#include <tensorflow_frontend/decoder.hpp>
#include <tensorflow_frontend/frontend.hpp>
//#include <tensorflow_frontend/place.hpp>
#include <vector>


//#include "node_context.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

//#define OPENVINO_VARIANT_DECLARATION(TYPE, info)                                          \
//    template <>                                                                           \
//    class ov::VariantWrapper<TYPE> : public ov::VariantImpl<TYPE> {                               \
//    public:                                                                               \
//        OPENVINO_RTTI(info);                                                              \
//        ov::VariantWrapper<TYPE>(const value_type& value) : ov::VariantImpl<value_type>(value) {} \
//    }
//
//namespace ov {
//OPENVINO_VARIANT_DECLARATION(int32_t, "Variant::int32");
//OPENVINO_VARIANT_DECLARATION(uint64_t, "Variant::uint64_t");
//OPENVINO_VARIANT_DECLARATION(std::vector<int32_t>, "Variant::int32_vector");
//OPENVINO_VARIANT_DECLARATION(float, "Variant::float");
//OPENVINO_VARIANT_DECLARATION(std::vector<float>, "Variant::float_vector");
//OPENVINO_VARIANT_DECLARATION(bool, "Variant::bool");
//OPENVINO_VARIANT_DECLARATION(ov::element::Type, "Variant::ov_element_type");
//OPENVINO_VARIANT_DECLARATION(std::vector<int64_t>, "Variant::int64_vector");
//OPENVINO_VARIANT_DECLARATION(ov::PartialShape, "Variant:ov_PartialShape");
//OPENVINO_VARIANT_DECLARATION(std::vector<std::string>, "Variant::string_vector");
////OPENVINO_VARIANT_DECLARATION(::tensorflow::DataType, "Variant::DataType");
////OPENVINO_VARIANT_DECLARATION(::tensorflow::TensorProto, "Variant::TensorProto");
//}  // namespace ov

//template <>
//class VariantWrapper<ov::element::Type> : public ov::VariantImpl<ov::element::Type> {
//public:
//    OPENVINO_RTTI("Variant::ov_element_type");
//    VariantWrapper(const value_type& value) : ov::VariantImpl<value_type>(value) {}
//};

namespace {
const std::map<::tensorflow::DataType, ov::element::Type>& TYPE_MAP() {
    static const std::map<::tensorflow::DataType, ov::element::Type> type_map{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16}};
    return type_map;
}
}  // namespace

//template <class T>
//bool is_type(const ngraph::VariantTypeInfo& type_info) {
//    return type_info == ngraph::VariantWrapper<T>::get_type_info_static();
//}

//template <class T>
//shared_ptr<ov::VariantWrapper<T>> create_variant(const T& data) {
//    return make_shared<ov::VariantWrapper<T>>(data);
//}

std::shared_ptr<ov::Variant> OVTFDecoder::get_attribute(const std::string& name,
                                                           const ngraph::VariantTypeInfo& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return nullptr;
    }

    std::cout << "OVTF_LOG - VariantTypeInfo - name: " << name << ", type_info: " << type_info << std::endl;
    //} else if (is_type<ov::element::Type>(type_info)) {
    std::cout << "OVTF_LOG - VariantTypeInfo - type_info.name: " << type_info.name << std::endl;
    if (strncmp(type_info.name, "Variant::ov_element_type", 24) == 0) {
        std::cout << "OVTF_LOG - VariantTypeInfo - ov::element::Type - A" << std::endl;
        auto data_type = attrs[0].type();
        //shared_ptr<ov::VariantWrapper<ov::element::Type>> vwrapper = create_variant<ov::element::Type>(TYPE_MAP().at(data_type));
        shared_ptr<ov::VariantWrapper<ov::element::Type>> vwrapper = make_shared<VariantWrapper<ov::element::Type>>(TYPE_MAP().at(data_type));
        //shared_ptr<ov::VariantWrapper<ov::element::Type>> vwrapper = make_shared<VariantWrapper<ov::element::Type>>(ov::element::f32);
        //shared_ptr<VariantWrapper<ov::element::Type>> vwrapper = make_shared<VariantWrapper<ov::element::Type>>(ov::element::f32);
        return vwrapper;
        //return ov::make_variant<ov::element::Type>(TYPE_MAP().at(data_type));
        //shared_ptr<ov::VariantImpl<ov::element::Type>> variant = make_shared<ov::VariantImpl<ov::element::Type>>(TYPE_MAP().at(data_type));
        //return variant;
    } else {
        std::cout << "OVTF_LOG - VariantTypeInfo - ov::element::Type - B" << std::endl;
    }
    return nullptr;
}

size_t OVTFDecoder::get_input_size() const {
    return m_input_size;
}

void OVTFDecoder::get_input_node(size_t input_port_idx,
                                    std::string& producer_name,
                                    size_t& producer_output_port_index) const {
}

const std::string& OVTFDecoder::get_op_type() const {
    return m_op_type;
}

const std::string& OVTFDecoder::get_op_name() const {
    return m_op_name;
}

void OVTFDecoder::set_next(shared_ptr<OVTFDecoder> next_ptr) {
    m_next_ptr = next_ptr;
}

shared_ptr<OVTFDecoder> OVTFDecoder::get_next() const {
    return m_next_ptr;
}

void OVTFDecoder::add_attr(std::string attr_name, ovtf_attr value) {
    m_attr_map.insert(pair<std::string, ovtf_attr>(attr_name, value));
}

void OVTFDecoder::set_tfnode(shared_ptr<tensorflow::Node> tfnode_ptr) {
    this->m_tfnode_ptr = tfnode_ptr;
}

vector<::tensorflow::AttrValue> OVTFDecoder::decode_attribute_helper(const string& name) const {
    auto attr_map = m_tfnode_ptr->def().attr();
    if (attr_map.contains(name))
        return {m_tfnode_ptr->def().attr().at(name)};
    return {};
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
