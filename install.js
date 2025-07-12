module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "echo app directory assumed to be 'app'"
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env", 
          path: "app",               // Edit this to customize the venv folder path
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python.exe -m pip install --upgrade pip",
          "uv pip install gradio==5.35.0"
        ]    
      }
    },
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install -U --pre triton-windows"
        ]
      }
    },
    {
      "when": "{{platform === 'darwin'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Install ffmpeg, libsndfile, cmake, and gcc (includes gfortran) via Homebrew, with Conda fallback for ffmpeg
          "brew install ffmpeg libsndfile cmake gcc || conda install ffmpeg -c conda-forge --yes || echo 'System dependencies (ffmpeg, libsndfile, cmake, gcc) installation failed. Please install manually with: brew install ffmpeg libsndfile cmake gcc'"
        ]
      }
    },
    {
      "when": "{{platform === 'linux'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Try apt-get for Debian/Ubuntu, yum for CentOS, with Conda fallback for ffmpeg
          "sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1-dev cmake gfortran || sudo yum install -y ffmpeg libsndfile-devel cmake gcc-gfortran || conda install ffmpeg -c conda-forge --yes || echo 'System dependencies (ffmpeg, libsndfile, cmake, gfortran) installation failed. Please install manually with: sudo apt-get install ffmpeg libsndfile1-dev cmake gfortran or sudo yum install ffmpeg libsndfile-devel cmake gcc-gfortran'"
        ]
      }
    },
    {
      "when": "{{platform === 'win32'}}",
      "method": "shell.run",
      "params": {
        "message": [
          // Windows only needs ffmpeg, as libsndfile, cmake, and gfortran are handled by Conda or not required
          "conda install ffmpeg -c conda-forge --yes || echo 'FFmpeg installation failed. Please install manually with: conda install ffmpeg'"
        ]
      }
    },
    // Step 7: Notify user
    {
      "method": "notify",
      "params": {
        "html": "Installation complete. Click the 'start' tab to launch StemXtract!"
      }
    }
  ]
};
 
