from util_for_argparse import get_cuda_port_args

args = get_cuda_port_args()

from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

# 导入 demo_run 中的方法
from projects.DensePose.demo_run import get_dp_predictor, get_outputs_by_predictor, create_context, execute_on_outputs

app = Flask(__name__)

# 初始化模型和上下文（只加载一次，节约内存和加速）
dp_cfg, dp_predictor = get_dp_predictor()
dp_context = create_context(dp_cfg)

@app.route('/densepose', methods=['POST'])
def densepose_process():
    # 允许用户传图片文件（POST multipart/form-data, form key: image）
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    try:
        img_pil = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    # DensePose inference
    dp_outputs = get_outputs_by_predictor(dp_predictor, img_pil)
    res_bg_dp, res_nobg_dp = execute_on_outputs(dp_context, dp_outputs, img_pil)
    res_bg_dp_pil = Image.fromarray(res_bg_dp)
    res_nobg_dp_pil = Image.fromarray(res_nobg_dp)

    # 将图片以二进制流形式回应成 file
    buf1 = BytesIO()
    buf2 = BytesIO()
    res_bg_dp_pil.save(buf1, format='PNG')
    res_nobg_dp_pil.save(buf2, format='PNG')
    buf1.seek(0)
    buf2.seek(0)

    # 可以返回文件或base64，这里用multi part形式返回
    # 使用 flask send_file 方式一个接口返回多个文件，需要定制: 比如打成zip或以JSON+base64返回
    # 下面实现为返回base64字符串（更易被API前端直接消费）
    import base64
    bg_img_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    nobg_img_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return jsonify({
        'bg_result_png_base64': bg_img_base64,
        'nobg_result_png_base64': nobg_img_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=False)
