import json
import urllib3
import base64

http = urllib3.PoolManager()
AI_SERVER_URL = "http://52.39.139.90:8000/segment"

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        image_b64 = body.get('image')
        
        if not image_b64:
            return {'statusCode': 400, 'body': json.dumps({'error': 'No image provided'})}

        image_bytes = base64.b64decode(image_b64)

        # 박스 좌표 추출
        params = []
        for key in ['box_x1', 'box_y1', 'box_x2', 'box_y2', 'point_x', 'point_y']:
            if key in body:
                params.append(f"{key}={body[key]}")
        
        url = AI_SERVER_URL
        if params:
            url += "?" + "&".join(params)

        response = http.request(
            'POST',
            url,
            fields={'file': ('image.png', image_bytes, 'image/png')}
        )
        
        return {
            'statusCode': response.status, 
            'body': response.data.decode('utf-8')
        }

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}