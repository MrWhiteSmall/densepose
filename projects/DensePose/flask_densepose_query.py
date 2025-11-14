from util_for_argparse import get_cuda_port_args

args = get_cuda_port_args()

import requests
import base64

def test_densepose_api(api_url=f"http://127.0.0.1:{args.port}/densepose", image_path="image.jpg"):
    """
    向DensePose API发送单张图片，并打印返回的结果Base64字符串长度做校验，并将bg_result结果保存为tmp.jpg。
    """
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(api_url, files=files)
    if response.status_code == 200:
        json_data = response.json()
        bg_base64_str = json_data.get("bg_result_png_base64", "")
        nobg_base64_str = json_data.get('nobg_result_png_base64',"")
        print("bg_result_png_base64长度:", len(bg_base64_str))
        print("nobg_result_png_base64长度:", len(nobg_base64_str))
        if bg_base64_str:
            img_bytes = base64.b64decode(bg_base64_str)
            with open("tmp.jpg", "wb") as out_img:
                out_img.write(img_bytes)
            print("已将bg_result保存到tmp.jpg")
        else:
            print("未收到bg_result_png_base64，未能保存图片。")
        if nobg_base64_str:
            nobg_image_bytes = base64.b64decode(nobg_base64_str)
            with open('tmp_nobg.jpg','wb') as f:
                f.write(nobg_image_bytes)
            print('save to nobg jpg')
        else:
            print('fail to save nobg jpg')
    else:
        print(f"请求失败: 状态码{response.status_code}，响应内容:{response.text}")


if __name__ == "__main__":
    test_densepose_api()
