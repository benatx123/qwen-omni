# Qwen Omni Deployment

**Version: 0.02**

This project provides a framework for deploying the [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) model using Transformers and Safetensors across multiple GPUs. The deployment is designed to be easily packaged into a Docker container for scalability and ease of use.

## Model Information

- **Model:** Qwen/Qwen2.5-Omni-7B
- **Source:** https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- **Description:** Qwen2.5-Omni-7B is a multi-modal model by Alibaba Group, capable of perceiving auditory and visual inputs, and generating text and speech. This repo demonstrates how to deploy it for inference with GPU support.

## Project Structure

```
qwen-omni-deployment
├── src
│   ├── app.py            # Main entry point for the application
│   ├── model_utils.py    # Utility functions for model handling
│   └── config.py         # Configuration settings for deployment
├── requirements.txt       # Python dependencies
├── Dockerfile             # Instructions for building the Docker image
├── .dockerignore          # Files to ignore when building the Docker image
├── README.md              # Project documentation
└── genai-stack.yaml       # Configuration for GenAI-Stack deployment
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd qwen-omni-deployment
   ```

2. **Install Dependencies**
   Ensure you have Python 3.8 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Model**
   Edit `src/config.py` to set the paths for model weights and configure GPU settings as needed.

4. **Run the Application**
   You can run the application directly using:
   ```bash
   python src/app.py
   ```

## Docker Deployment

To build and run the Docker container, follow these steps:

1. **Build the Docker Image**
   ```bash
   docker build -t qwen-omni .
   ```

2. **Run the Docker Container**
   ```bash
   docker run --gpus all -p 5000:5000 qwen-omni
   ```

## Usage

Once the application is running, you can send requests for inference to the model. The web UI at `/` provides a simple chat interface. While the model is generating a response, a spinning wheel is shown. After completion, you will see the model's response along with metrics:

- **Response time (ms)**
- **Tokens per second**

You can also POST to `/infer` with a JSON body containing a `conversation` key (see `src/app.py` for details).

## GenAI-Stack Deployment

To deploy the application using GenAI-Stack, configure the `genai-stack.yaml` file with the necessary resources and environment variables, then follow the GenAI-Stack deployment instructions.

## FAQ

**Q: Where can I find the Qwen2.5-Omni-7B model?**
A: [https://huggingface.co/Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

**Q: What metrics are reported?**
A: The API and web UI report response time (ms) and tokens per second for each inference.

**Q: How do I use multiple GPUs?**
A: The deployment uses `device_map="auto"` to automatically distribute the model across available GPUs.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.