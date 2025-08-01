// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Gemm operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunGemmTest(const std::vector<TestInputDef<DataType>>& input_defs,
                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& backend_name = "cpu",
                        int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("Gemm", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Gemm with non-default 'alpha' or 'beta' attributes is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Gemm_NonDefaultAlphaBeta_Unsupported) {
  // Check that alpha != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f)},
                     {utils::MakeAttribute("alpha", 1.5f)},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.

  // Check that beta != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                     {utils::MakeAttribute("beta", 1.2f)},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Gemm with 2D bias is supported.
TEST_F(QnnCPUBackendTests, Gemm_2D_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 12);

  // 2D matrix mul with bias is supported.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, input_a_data),
                      TestInputDef<float>({3, 4}, false, input_b_data),
                      TestInputDef<float>({2, 4}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All);  // Assigned to QNN EP.

  // However, 2D matrix mul without a bias is supported. Input A's 0th dimension is interpreted as `batch_size`.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, input_a_data),
                      TestInputDef<float>({3, 4}, false, input_b_data)},
                     {},
                     ExpectedEPNodeAssignment::All);  // Assigned to QNN EP.
}

// since Qnn v2.34 value pair (120.73912, 121.73912) at index #0 don't match, which is 1 from 120.739
// Test Gemm with dynamic (i.e., not initializer) inputs (A, B, Bias).
TEST_F(QnnCPUBackendTests, DISABLED_Gemm_Dynamic_A_B_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

// Test Gemm with static B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

// Test Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_TransAB_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                      utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All);
}

// Since Qnn 2.34 value pair (29.4347763, 30.4347763) at index #0 don't match, which is 1 from 29.4348
// Test Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnCPUBackendTests, DISABLED_Gemm_TransAB_Dynamic_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                      utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All);
}

// Since Qnn 2.34 value pair (11, 10) at index #0 don't match, which is -1 from 11
TEST_F(QnnCPUBackendTests, DISABLED_Gemm_Broadcast_Bias_DynamicInputs) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // All dynamic inputs
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, false, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

// TODO: When this is fixed, enable GemmOpTypedTests/0.TestGemmBroadcast test in cpu/math/gemm_test.cc
// This began failing in QNN SDK 2.17 for the CPU backend.
// Log: the value pair (11, 10) at index #0 don't match, which is -1 from 11
TEST_F(QnnCPUBackendTests, DISABLED_Gemm_Broadcast_Bias_DynamicA_StaticB_DynamicC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, dynamic C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_StaticC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, static C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

namespace {
GetTestModelFn BuildReshapeGemmTestCase(const TestInputDef<float>& input, const TestInputDef<int64_t>& shape,
                                        const TestInputDef<float>& weight, const TestInputDef<float>& bias) {
  return [&](ModelTestBuilder& builder) {
    std::vector<NodeArg*> reshape_inputs = {MakeTestInput<float>(builder, input),
                                            MakeTestInput<int64_t>(builder, shape)};
    auto* reshape_output = builder.MakeIntermediate();
    builder.AddNode("Reshape", reshape_inputs, {reshape_output});
    NodeArg* output = builder.MakeOutput();
    std::vector<NodeArg*> gemm_inputs = {reshape_output, MakeTestInput<float>(builder, weight),
                                         MakeTestInput<float>(builder, bias)};
    builder.AddNode("Gemm", gemm_inputs, {output});
  };
}

void RunReshapeGemmTest(const TestInputDef<float>& input, const TestInputDef<int64_t>& shape,
                        const TestInputDef<float>& weight, const TestInputDef<float>& bias,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& backend_name = "cpu", float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  auto build_fn = BuildReshapeGemmTestCase(input, shape, weight, bias);
  RunQnnModelTest(build_fn, provider_options, 18, expected_ep_assignment, fp32_abs_err);
}

}  // namespace

TEST_F(QnnCPUBackendTests, ReshapeGemmFusion) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> shape_data = {4, 2};
  std::vector<float> weight_data(6, 1.0f);
  std::vector<float> bias_data = {1.0f, 2.0f, 3.0f};
  RunReshapeGemmTest(TestInputDef<float>({2, 2, 2}, false, input_data), TestInputDef<int64_t>({2}, true, shape_data),
                     TestInputDef<float>({2, 3}, true, weight_data), TestInputDef<float>({3}, true, bias_data),
                     ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that builds a model with a QDQ Gemm node.
template <typename InputAQType, typename InputBQType>
inline GetTestQDQModelFn<InputAQType> BuildQDQGemmTestCase(const std::vector<TestInputDef<float>>& input_defs,
                                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                           bool use_contrib_qdq = false) {
  return [input_defs, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                              std::vector<QuantParams<InputAQType>>& output_qparams) {
    const size_t num_inputs = input_defs.size();
    assert(num_inputs == 2 || num_inputs == 3);

    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(num_inputs);

    // Process input 0
    NodeArg* input0 = MakeTestInput<float>(builder, input_defs[0]);
    QuantParams<InputAQType> input0_qparams = GetTestInputQuantParams<InputAQType>(input_defs[0]);
    NodeArg* input0_after_qdq = AddQDQNodePair<InputAQType>(builder, input0, input0_qparams.scale,
                                                            input0_qparams.zero_point, use_contrib_qdq);
    op_inputs.push_back(input0_after_qdq);

    // Process input 1
    NodeArg* input1 = MakeTestInput<float>(builder, input_defs[1]);
    QuantParams<InputBQType> input1_qparams = GetTestInputQuantParams<InputBQType>(input_defs[1]);
    NodeArg* input1_after_qdq = AddQDQNodePair<InputBQType>(builder, input1, input1_qparams.scale,
                                                            input1_qparams.zero_point, use_contrib_qdq);
    op_inputs.push_back(input1_after_qdq);

    // Process bias
    if (num_inputs == 3) {
      NodeArg* bias_input = MakeTestQDQBiasInput(builder, input_defs[2], input0_qparams.scale * input1_qparams.scale,
                                                 use_contrib_qdq);
      op_inputs.push_back(bias_input);
    }

    // Op -> op_output
    auto* gemm_output = builder.MakeIntermediate();
    Node& gemm_node = builder.AddNode("Gemm", op_inputs, {gemm_output});

    for (const auto& attr : attrs) {
      gemm_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputAQType>(builder, gemm_output, output_qparams[0].scale,
                                                       output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a QDQ Gemm model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename InputAQType, typename InputBQType>
static void RunQDQGemmTestOnHTP(const std::vector<TestInputDef<float>>& input_defs,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13,
                                bool use_contrib_qdq = false,
                                QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float>("Gemm", input_defs, {}, attrs);
  auto qdq_model_builder = BuildQDQGemmTestCase<InputAQType, InputBQType>(input_defs, attrs, use_contrib_qdq);
  TestQDQModelAccuracy<InputAQType>(f32_model_builder,
                                    qdq_model_builder,
                                    provider_options,
                                    opset,
                                    expected_ep_assignment,
                                    tolerance);
}

// Test 8-bit QDQ Gemm with dynamic inputs A and Bias. The B input is an initializer.
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Static_B_Dynamic_Bias_U8) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, false, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All);
}

#ifndef __linux__
// Test 16-bit QDQ Gemm with dynamic inputs A and Bias. The B input is an initializer.
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Dynamic_B_Dynamic_Bias_U16) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint16_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                           TestInputDef<float>({6, 4}, false, input_b_data),
                                           TestInputDef<float>({1, 4}, false, input_c_data)},
                                          {},
                                          ExpectedEPNodeAssignment::All,
                                          13,     // opset
                                          true);  // Use com.microsoft Q/DQ ops
}
#endif

// Test broadcasting of bias input. All inputs are dynamic.
TEST_F(QnnHTPBackendTests, Gemm_Broadcast_Bias_DynamicInputs) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // All dynamic inputs
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({2, 4}, false, input_a_data),
                                         TestInputDef<float>({4, 3}, false, input_b_data),
                                         TestInputDef<float>({3}, false, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All,
                                        13,
                                        false,
                                        QDQTolerance(0.00410f));
}

TEST_F(QnnHTPBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_DynamicC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, dynamic C
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({2, 4}, false, input_a_data),
                                         TestInputDef<float>({4, 3}, true, input_b_data),
                                         TestInputDef<float>({3}, false, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All,
                                        13,
                                        false,
                                        QDQTolerance(0.00410f));
}

TEST_F(QnnHTPBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_StaticC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, static C
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({2, 4}, false, input_a_data),
                                         TestInputDef<float>({4, 3}, true, input_b_data),
                                         TestInputDef<float>({3}, true, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All,
                                        13,
                                        false,
                                        QDQTolerance(0.00410f));
}

// Test 16-bit QDQ Gemm with dynamic inputs A and Bias. The B input is an initializer.
// TODO: Inaccuracy detected for output 'output_0', element 0.
// Output quant params: scale=0.001872879103757441, zero_point=0.
// Expected val: 120.73912048339844
// QNN QDQ val: 0 (err 120.73912048339844)
// CPU QDQ val: 120.73889923095703 (err 0.00022125244140625)
// Issue fixed in 2.30
#ifdef __linux__
// Failed on Linux with 2.31
TEST_F(QnnHTPBackendTests, DISABLED_Gemm_Dynamic_A_Static_B_Dynamic_Bias_U16) {
#else
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Static_B_Dynamic_Bias_U16) {
#endif
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint16_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                           TestInputDef<float>({6, 4}, true, input_b_data),
                                           TestInputDef<float>({1, 4}, false, input_c_data)},
                                          {},
                                          ExpectedEPNodeAssignment::All,
                                          13,     // opset
                                          true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Gemm (16bit act, 8bit weight) with dynamic inputs A and Bias. The B input is an initializer.
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Static_B_Dynamic_Bias_U16Act_U8Weight) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                          TestInputDef<float>({6, 4}, true, input_b_data),
                                          TestInputDef<float>({1, 4}, false, input_c_data)},
                                         {},
                                         ExpectedEPNodeAssignment::All,
                                         13,     // opset
                                         true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Gemm with dynamic A and B inputs. The Bias is static.
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.48132994771003723, zero_point=0.
// Expected val: 120.73912048339844
// QNN QDQ val: 77.012794494628906 (err 43.726325988769531)
// CPU QDQ val: 119.85115814208984 (err 0.88796234130859375)
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_B_Static_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, false, input_b_data),  // Dynamic => inaccuracy
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Gemm with static B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_TransAB_Static_B_And_Bias_U8) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({6, 1}, false, input_a_data),
                                         TestInputDef<float>({4, 6}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                                         utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Gemm (16bit activation, 8bit weight) with transposed A/B and static B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_TransAB_Static_B_And_Bias_U16Act_U8Weight) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint8_t>({TestInputDef<float>({6, 1}, false, input_a_data),
                                          TestInputDef<float>({4, 6}, true, input_b_data),
                                          TestInputDef<float>({1, 4}, true, input_c_data)},
                                         {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                                          utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                                         ExpectedEPNodeAssignment::All,
                                         13,     // opset
                                         true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_TransAB_Dynamic_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({6, 1}, false, input_a_data),
                                         TestInputDef<float>({4, 6}, false, input_b_data),
                                         TestInputDef<float>({1, 4}, false, input_c_data)},
                                        {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                                         utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// Gemm basic default attributes.
// QNN's FullyConnected operator only supports `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
// Input A's 0th dimension is interpreted as `batch_size`.
TEST_F(QnnGPUBackendTests, Gemm_Basic) {
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, -10.0f, 10.0f),
                      TestInputDef<float>({3, 4}, false, -10.0f, 10.0f)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Gemm with 'alpha' or 'beta' attributes is not supported by QNN EP.
TEST_F(QnnGPUBackendTests, Gemm_AlphaBetaUnsupported) {
  // Check that alpha != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f)},
                     {utils::MakeAttribute("alpha", 1.5f)},
                     ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                     "gpu");

  // Check that beta != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                     {utils::MakeAttribute("beta", 1.2f)},
                     ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                     "gpu");
}

// Gemm with matrix bias ie 2D (M, N) is supported.
// When vector bias ie M == 1
// QNN's FullyConnected operator only supports `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
// When 2D bias i.e. M != 1, N != 1.
// When 2D bias i.e. M != 1, N != 1.
// QNN's Gemm will be split in to FullyConnected and ElementwiseAdd.
TEST_F(QnnGPUBackendTests, Gemm_2D_Bias) {
  // 2D matrix mul with 2D bias is supported when Gemm is not a QDQ node.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, -10.0f, 10.0f),
                      TestInputDef<float>({3, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All,  // Should be assigned to QNN EP.
                     "gpu");
}

// Gemm with vector bias is supported ie when M == 1.
// Bias is broadcast across input batches.
// `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
TEST_F(QnnGPUBackendTests, Gemm_1DBiasBcast) {
  // 2D matrix mul with 1D bias supported.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, -10.0f, 10.0f),
                      TestInputDef<float>({3, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with dynamic (i.e., not initializer) inputs (A, B, Bias).
TEST_F(QnnGPUBackendTests, Gemm_Dynamic_A_B_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with static B and Bias inputs.
TEST_F(QnnGPUBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnGPUBackendTests, Gemm_TransposeAB_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                      utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnGPUBackendTests, Gemm_TransAB_Dynamic_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {utils::MakeAttribute("transA", static_cast<int64_t>(1)),
                      utils::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Bias broadcast across batches.
TEST_F(QnnGPUBackendTests, Gemm_Broadcast_Bias_DynamicInputs) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};

  // All dynamic inputs
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, false, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

TEST_F(QnnGPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_DynamicC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};

  // Dynamic A, static B, dynamic C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

TEST_F(QnnGPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_StaticC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};

  // Dynamic A, static B, static C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Tests fusion of Reshape inpout followed by Gemm.
TEST_F(QnnGPUBackendTests, ReshapeGemmFusion) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> shape_data = {4, 2};
  std::vector<float> weight_data(6, 1.0f);
  std::vector<float> bias_data = {1.0f, 2.0f, 3.0f};
  RunReshapeGemmTest(TestInputDef<float>({2, 2, 2}, false, input_data), TestInputDef<int64_t>({2}, true, shape_data),
                     TestInputDef<float>({2, 3}, true, weight_data), TestInputDef<float>({3}, true, bias_data),
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
