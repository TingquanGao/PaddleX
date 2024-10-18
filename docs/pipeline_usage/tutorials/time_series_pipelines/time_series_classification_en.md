[简体中文](time_series_classification.md) | English

# Time Series Classification Pipeline Tutorial

## 1. Introduction to General Time Series Classification Pipeline
Time series classification is a technique that categorizes time-series data into predefined classes, widely applied in fields such as behavior recognition and financial trend analysis. By analyzing features that vary over time, it identifies different patterns or events, for example, classifying a speech signal as "greeting" or "request," or categorizing stock price movements as "rising" or "falling." Time series classification typically employs machine learning and deep learning models, effectively capturing temporal dependencies and variation patterns to provide accurate classification labels for data. This technology plays a pivotal role in applications such as intelligent monitoring and market forecasting.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/01.png)

**The General Time Series Classification Pipeline includes a Time Series Classification module.**

<details>
   <summary> 👉Model List Details</summary>

|Model Name|Acc(%)|Model Size (M)|
|-|-|-|
|TimesNet_cls|87.5|792K|

**Note: The above accuracy metrics are measured on the [UWaveGestureLibrary](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv) dataset. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
PaddleX provides pre-trained model pipelines that can be quickly experienced. You can experience the effects of the General Time Series Classification Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/105707/webUI?source=appCenter) the effects of the General Time Series Classification Pipeline using the official demo for recognition, for example:

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/02.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the model in the pipeline online**.

Note: Due to the close relationship between time series data and scenarios, the official built-in model for online experience of time series tasks is only a model solution for a specific scenario and is not a general solution applicable to other scenarios. Therefore, the experience method does not support using arbitrary files to experience the effect of the official model solution. However, after training a model for your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience
Before using the General Time Series Classification Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

#### 2.2.1 Command Line Experience
Experience the effects of the time series classification pipeline quickly with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the time series classification pipeline.
--input: The local path or URL of the input sequence to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default time series classification pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to Expand</summary>

```bash
paddlex --get_pipeline_yaml ts_cls
```
After execution, the time series classification pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config ts_cls --save_path ./my_path
```

After obtaining the pipeline configuration file, you can replace `--pipeline` with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is `./ts_ad.yaml`, simply execute:

```bash
paddlex --pipeline ./ts_cls.yaml --input ts_cls.csv --device gpu:0
```

In this command, parameters such as `--model` and `--device` are not required to be specified, as they will use the parameters defined in the configuration file. If these parameters are specified, the specified values will take precedence.

</details>

After execution, the result is:

```bash
{'input_path': 'ts_cls.csv', 'classification':         classid     score
sample
0             0  0.617688}
```

#### 2.2.2 Python Script Integration

A few lines of code can complete rapid inference for production lines. Taking the General Time Series Classification Pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_cls")

output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_csv("./output/")  # Save results in CSV format
```

The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX. | `str` | None |
| `device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `use_hpip` | Whether to enable high-performance inference. Available only if the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the pipeline object for inference: The `predict` method takes `x` as a parameter, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| `str` | Supports passing the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv). |
| `str` | Supports passing a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing a dictionary type, where the key needs to correspond to the specific task, e.g., "img" for image classification tasks, and the value of the dictionary supports the above data types, e.g., `{"img": "/root/data1"}`. |
| `list` | Supports passing a list, where the list elements need to be of the above types, such as `[numpy.ndarray, numpy.ndarray]`, `["/root/data/img1.jpg", "/root/data/img2.jpg"]`, `["/root/data1", "/root/data2"]`, `[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained by iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/ts_cls.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_cls.yaml")
output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_csv("./output/")  # Save results in CSV format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for deployment performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that deeply optimize model inference and pre/post-processing to significantly speed up the end-to-end process. Refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference_en.md) for detailed high-performance inference procedures.

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX enables users to achieve low-cost service-oriented deployment of pipelines. Refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md) for detailed service-oriented deployment procedures.

Below are the API references and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- Both the response body and the request body for POST requests are JSON data (JSON objects).
- When the request is processed successfully, the response status code is `200`, and the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Fixed as `0`.|
    |`errorMsg`|`string`|Error message. Fixed as `"Success"`.|

    The response body may also have a `result` property of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Same as the response status code.|
    |`errorMsg`|`string`|Error message.|

Operations provided by the service:

- **`infer`**

    Classify time-series data.

    `POST /time-series-classification`

    - The request body properties are as follows:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        |`csv`|`string`|The URL of a CSV file accessible by the service or the Base64 encoded result of the CSV file content. The CSV file must be encoded in UTF-8.|Yes|

    - When the request is processed successfully, the `result` in the response body has the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        |`label`|`string`|Class label.|
        |`score`|`number`|Class score.|

        An example of `result` is as follows:

        ```json
        {
          "label": "running",
          "score": 0.97
        }
        ```

</details>

<details>
<summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/time-series-classification"
csv_path = "./test.csv"

with open(csv_path, "rb") as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode("ascii")

payload = {"csv": csv_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
print(f"label: {result['label']}, score: {result['score']}")
```

</details>

<details>
<summary>C++</summary>

```cpp
#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string csvPath = "./test.csv";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    std::ifstream file(csvPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedCsv = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["csv"] = encodedCsv;
    std::string body = jsonObj.dump();

    auto response = client.Post("/time-series-classification", headers, body, "application/json");
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];
        std::cout << "label: " << result["label"] << ", score: " << result["score"] << std::endl;
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        std::cout << response->body << std::endl;
        return 1;
    }

    return 0;
}
```

</details>

<details>
<summary>Java</summary>

```java
import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/time-series-classification";
        String csvPath = "./test.csv";

        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("csv", csvData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                System.out.println("label: " + result.get("label").asText() + ", score: " + result.get("score").asText());
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
```

</details>

<details>
<summary>Go</summary>

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/time-series-classification"
    csvPath := "./test.csv";

    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println("Error reading csv file:", err)
        return
    }
    csvData := base64.StdEncoding.EncodeToString(csvBytes)

    payload := map[string]string{"csv": csvData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer res.Body.Close()

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Label string `json:"label"`
            Score string `json:"score"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    fmt.Printf("label: %s, score: %s\n", respData.Result.Label, respData.Result.Score)
}
```

</details>

<details>
<summary>C#</summary>

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/time-series-classification";
    static readonly string csvPath = "./test.csv";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] csveBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csveBytes);

        var payload = new JObject{ { "csv", csvData } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string label = jsonResponse["result"]["label"].ToString();
        string score = jsonResponse["result"]["score"].ToString();
        Console.WriteLine($"label: {label}, score: {score}");
    }
}
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-classification'
const csvPath = "./test.csv";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'csv': encodeFileToBase64(csvPath)
  })
};

function encodeFileToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) => {
    const result = response.data["result"];
    console.log(`label: ${result["label"]}, score: ${result["score"]}`);
})
.catch((error) => {
  console.log(error);
});
```

</details>

<details>
<summary>PHP</summary>

```php
<?php

$API_URL = "http://localhost:8080/time-series-classification";
$csv_path = "./test.csv";

$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array("csv" => $csv_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"];
echo "label: " . $result["label"] . ", score: " . $result["score"];

?>
```

</details>
</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. Refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy.md) for detailed edge deployment procedures.
Choose the appropriate deployment method based on your needs to proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Time Series Classification Pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Time Series Classification Pipeline includes a time series classification module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/time_series_modules/time_series_classification_en.md#iv-custom-development) section in the [Time Series Classification Module Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_classification_en.md) to fine-tune the time series classification model using your private dataset.

### 4.2 Model Application
After fine-tuning the model with your private dataset, you will obtain local model weights.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: TimesNet_cls  # Replace with the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Then, refer to the command line or Python script methods in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for time series classification pipeline inference, the Python command is:

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` flag in the Python command as follows:

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device npu:0
```

If you intend to use the General Time Series Classification Pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide_en.md).
