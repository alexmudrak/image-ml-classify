<p align="center">
    <img width="300" height="300" src="static/logo.png" />
</p>

# Image ML Classify - Image Processing and ResNet-18 Model Training Tool

**Image ML Classify** is a versatile web application that serves as a powerful image processing and **ResNet-18** model training tool. It provides the convenience of storing and synchronizing datasets in the **Yandex Disk** cloud storage.

The application utilizes Docker for containerization, and model training occurs in the background, freeing the main thread for other tasks. All datasets can be stored and synchronized with **Yandex Disk**, ensuring data availability and convenience for model training.

# Endpoints

Below are the endpoints and sample CURL requests for using this tool:

## Image Classification

Upload an image and obtain classification results using the pre-trained ResNet-18 model.

### CURL Request:
```bash
curl -X POST \
-F "file=@your_image.jpg" \
http://your-api-url/api/v1/image
```

## Model Training
Initiate the training of the ResNet-18 model with your custom datasets in the background.

### CURL Request (GET - Get Current Training Status):
```bash
curl -X GET \
-H "X-Key: your_api_key" \
http://your-api-url/api/v1/train
```

### CURL Request (POST - Start Training):
```bash
curl -X POST \
-H "X-Key: your_api_key" \
-H "Content-Type: application/json" \
-d '{"epoch": 10, "hard_run": false, "run_as": "all"}' \
http://your-api-url/api/v1/train
```



# Getting credentials for Yandex.Disk
<details>
    <summary>How-To?</summary>
Requirements:

- Create a new application (optional, you can use a third-party
application; you'll need its ID and secret)
    - App ID
    - App Secret
- Authorize the application to access the disk
    - Get the authorization code
    - Get the access token

## Create an Application (Optional)
Official instructions - `https://yandex.ru/dev/disk/api/concepts/quickstart.html#quickstart__oauth`

The main goal is to obtain the App ID and App Secret; you can use an existing
application.

## Obtaining an Authorization Code

Navigate to the following link - `https://oauth.yandex.ru/authorize?response_type=code&client_id={APP_ID}`
You will be asked to confirm permissions for the application.

After redirection, you will be provided with a code `http://.../?code=7122172`
*the authorization code is valid for 15 minutes*

## Obtaining a Token

After obtaining the authorization code, you can acquire the application's
authorization token. To do this, make a `POST` request, providing the necessary
data

```bash
curl -X POST \
-L "https://oauth.yandex.ru/token" \
-F "grant_type=authorization_code" \
-F "client_id={APP_ID}" \
-F "client_secret={APP_SECRET}" \
-F "code={AUTH_CODE}" \
```

</details>
