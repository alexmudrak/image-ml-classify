# Run

-   Need to add base model into `datamodels` folder



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
