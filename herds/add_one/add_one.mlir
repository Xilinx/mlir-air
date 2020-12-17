module {
  func @graph(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = "aten.constant"() {type = "f32", value = 1.000000e+00 : f32} : () -> f32
    %1 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %2 = "aten.add"(%arg0, %0, %1) {acdc_layer_name = "L0-add-0"} : (tensor<256x256xf32>, f32, i32) -> tensor<256x256xf32>
    return %2 : tensor<256x256xf32>
  }
}
