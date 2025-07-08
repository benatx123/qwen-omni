# Qwen Omni Deployment

This project provides a framework for deploying the Qwen/Qwen2.5-Omni-7B model using Transformers and Safetensors across multiple GPUs. The deployment is designed to be easily packaged into a Docker container for scalability and ease of use.

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

Once the application is running, you can send requests for inference to the model. Refer to the documentation in `src/app.py` for details on the API endpoints and request formats.

## GenAI-Stack Deployment

To deploy the application using GenAI-Stack, configure the `genai-stack.yaml` file with the necessary resources and environment variables, then follow the GenAI-Stack deployment instructions.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.