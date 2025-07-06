import time
import streamlit as st
import docker
import cv2
import subprocess
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize Docker client
try:
    client = docker.from_env()
except Exception as e:
    st.error(f"Error initializing Docker client: {e}")
    client = None

# Streamlit page setup
st.set_page_config(page_title="Docker AI Automation Dashboard", layout="wide")
st.title("🐳 CHAL DOCKERRR CHALE")

menu = st.sidebar.radio("📋 Choose Feature", [
    "Basic Docker Commands",
    "Launch GUI Apps & Jenkins",
    "Launch LR Model with Flask API",
    "Build from Dockerfile",
    "Commit Container to Image",
    "Docker-in-Docker (DinD)",
    "Deploy Apache Webserver in Docker",
    "📘 Docker Knowledge Section",
    "📖 Blog: Why Big Boiiisss Use Docker",
    "📺 Emotion Detection (Webcam)",
    "Exit"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by **Deepali Verma**")

def run_shell_command(command):
    """Run shell command and return (success, output)"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out."
    except Exception as e:
        return False, str(e)

def list_containers():
    if not client:
        st.error("Docker client not initialized.")
        return
    containers = client.containers.list(all=True)
    if containers:
        data = []
        for c in containers:
            data.append({"Name": c.name, "ID": c.short_id, "Status": c.status, "Image": c.image.tags[0] if c.image.tags else "unknown"})
        st.table(data)
    else:
        st.info("No containers found.")

def list_images():
    if not client:
        st.error("Docker client not initialized.")
        return
    images = client.images.list()
    if images:
        data = []
        for img in images:
            tags = img.tags[0] if img.tags else "<none>"
            data.append({"ID": img.short_id, "Tags": tags, "Size (MB)": round(img.attrs['Size']/(1024*1024), 2)})
        st.table(data)
    else:
        st.info("No images found.")

# Emotion Detector class
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img
# Apache Webserver Setup
if menu == "Deploy Apache Webserver in Docker":
    st.header("🚀 Deploy Apache Webserver in Docker")
    st.write("This feature sets up an Apache webserver inside a Docker container.")

    if st.button("Deploy Apache Webserver"):
        apache_dockerfile = """FROM httpd:2.4
COPY ./public-html/ /usr/local/apache2/htdocs/"""
        try:
            import os
            os.makedirs("apache/public-html", exist_ok=True)
            with open("apache/public-html/index.html", "w") as f:
                f.write("<h1>Hello from Apache inside Docker!</h1>")
            with open("apache/Dockerfile", "w") as f:
                f.write(apache_dockerfile)
            success, build_output = run_shell_command("docker build -t apache-webserver ./apache")
            if success:
                run_success, run_output = run_shell_command("docker run -d -p 8088:80 --name apache-server apache-webserver")
                if run_success:
                    st.success("Apache Webserver is running at http://localhost:8088")
                else:
                    st.error(run_output)
            else:
                st.error(build_output)
        except Exception as e:
            st.error(f"Error during setup: {e}")

# Blog Section
if menu == "📖 Blog: Why Big Boiiisss Use Docker":
    st.title("📖 Why the Big Boiiisss Use Docker 🐳")

    st.markdown("""
Welcome to the world of containers! Here’s why tech giants—oops, we mean **Big Boiiisss** like Tesla, Google, Netflix, and others can’t stop raving about Docker:

---

### 🚗 Tesla
- Uses Docker to test and deploy autonomous systems.
- Enables fast iteration on AI models for vehicles.

### 🎵 Spotify
- Runs containerized backend services for music streaming.
- Scales instantly based on user demand.

### 🎮 Google
- Manages huge Kubernetes clusters powered by Docker.
- Ensures every service is consistent and isolated.
- Enhanced developer productivity with containerized dev environments.

### 🎮 Netflix
- Manages thousands of microservices with Docker.
- Reduces deployment time to seconds.
- Makes A/B testing and service updates super smooth.

### 💰 PayPal
- Accelerated their CI/CD pipelines.
- Reduced infrastructure cost by using containers.
- Improved service uptime and rollback strategies.

### 🤬 NASA
- Uses Docker to simulate mission-critical systems.
- Containers help with reproducibility and high-performance computing.

### 🪖 General Benefits
- **🚨 Speed**: Spin up dev environments and deploy apps in seconds.
- **🔄 Portability**: Run the same app on your laptop, staging, and production.
- **🔐 Isolation**: Separate apps mean fewer bugs and better security.
- **🛠️ Automation**: Perfect for CI/CD and Infrastructure as Code.
- **💡 Scalability**: Add or remove containers based on load.
- **🚀 Lightweight**: No need for heavy VMs — containers share the host kernel.

> **TL;DR:** Docker helps big companies move faster, break fewer things, and build epic stuff faster.

**Moral of the story?** If you're still deploying with zip files and FTP, maybe it's time to **DOCKERIZE YOUR LIFE**. 🚫🚚💪
    """)

# ------------------------------------
# Main menu logic
# ------------------------------------

if menu == "Basic Docker Commands":
    action = st.selectbox("Select Command", [
        "List All Containers", "List All Images", "Launch New Container",
        "Start Container", "Stop Container", "Remove Container",
        "Pull Image", "Remove Image", "Remove All Images"
    ])

    if action == "List All Containers":
        list_containers()

    elif action == "List All Images":
        list_images()

    elif action == "Launch New Container":
        with st.form("launch_container_form"):
            name = st.text_input("Container Name", key="launch_name")
            image = st.text_input("Image Name", "ubuntu", key="launch_image")
            submit = st.form_submit_button("Launch")
            if submit:
                if not name.strip() or not image.strip():
                    st.error("Container name and image name cannot be empty.")
                else:
                    cmd = f"docker run -dit --name {name.strip()} {image.strip()}"
                    success, output = run_shell_command(cmd)
                    if success:
                        st.success(f"Container '{name}' launched successfully.\n{output}")
                    else:
                        st.error(f"Failed to launch container:\n{output}")

    elif action == "Start Container":
        with st.form("start_container_form"):
            name = st.text_input("Container Name to Start", key="start_name")
            submit = st.form_submit_button("Start")
            if submit:
                if not name.strip():
                    st.error("Container name cannot be empty.")
                else:
                    try:
                        container = client.containers.get(name.strip())
                        container.start()
                        st.success(f"Container '{name}' started successfully.")
                    except docker.errors.NotFound:
                        st.error(f"No container found with name '{name}'.")
                    except Exception as e:
                        st.error(f"Error starting container: {e}")

    elif action == "Stop Container":
        with st.form("stop_container_form"):
            name = st.text_input("Container Name to Stop", key="stop_name")
            submit = st.form_submit_button("Stop")
            if submit:
                if not name.strip():
                    st.error("Container name cannot be empty.")
                else:
                    try:
                        container = client.containers.get(name.strip())
                        container.stop()
                        st.success(f"Container '{name}' stopped successfully.")
                    except docker.errors.NotFound:
                        st.error(f"No container found with name '{name}'.")
                    except Exception as e:
                        st.error(f"Error stopping container: {e}")

    elif action == "Remove Container":
        with st.form("remove_container_form"):
            name = st.text_input("Container Name to Remove", key="remove_name")
            submit = st.form_submit_button("Remove")
            if submit:
                if not name.strip():
                    st.error("Container name cannot be empty.")
                else:
                    try:
                        container = client.containers.get(name.strip())
                        container.remove(force=True)
                        st.success(f"Container '{name}' removed successfully.")
                    except docker.errors.NotFound:
                        st.error(f"No container found with name '{name}'.")
                    except Exception as e:
                        st.error(f"Error removing container: {e}")

    elif action == "Pull Image":
        with st.form("pull_image_form"):
            image = st.text_input("Image to Pull", "ubuntu:latest", key="pull_image")
            submit = st.form_submit_button("Pull")
            if submit:
                if not image.strip():
                    st.error("Image name cannot be empty.")
                else:
                    success, output = run_shell_command(f"docker pull {image.strip()}")
                    if success:
                        st.success(f"Image '{image}' pulled successfully.\n{output}")
                    else:
                        st.error(f"Failed to pull image:\n{output}")

    elif action == "Remove Image":
        with st.form("remove_image_form"):
            image = st.text_input("Image to Remove", key="remove_image")
            submit = st.form_submit_button("Remove")
            if submit:
                if not image.strip():
                    st.error("Image name cannot be empty.")
                else:
                    try:
                        client.images.remove(image.strip())
                        st.success(f"Image '{image}' removed successfully.")
                    except docker.errors.ImageNotFound:
                        st.error(f"No image found with name '{image}'.")
                    except Exception as e:
                        st.error(f"Error removing image: {e}")

    elif action == "Remove All Images":
        if st.button("Remove All Images"):
            confirm = st.checkbox("Confirm removing ALL images (cannot be undone)")
            if confirm:
                try:
                    images = client.images.list()
                    for img in images:
                        try:
                            client.images.remove(img.id, force=True)
                        except Exception as e:
                            st.warning(f"Could not remove image {img.id}: {e}")
                    st.success("All images removed.")
                except Exception as e:
                    st.error(f"Error removing images: {e}")
            else:
                st.info("Please confirm removal by checking the box.")

elif menu == "Launch GUI Apps & Jenkins":
    apps = {
        "VLC": "jlesage/vlc",
        "Firefox": "jlesage/firefox",
        "Jenkins": "jenkins/jenkins"
    }

    for name, image in apps.items():
        st.subheader(f"🚀 {name}")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"Launch {name}"):
                ports = ""
                if name == "Jenkins":
                    ports = "-p 8080:8080 -p 50000:50000"
                try:
                    existing = client.containers.get(name.lower())
                    st.warning(f"Container {name.lower()} already exists. Removing before launching.")
                    existing.remove(force=True)
                except docker.errors.NotFound:
                    pass
                cmd = f"docker run -d {ports} --name {name.lower()} --privileged {image}"
                success, output = run_shell_command(cmd)
                if success:
                    st.success(f"{name} container launched.")
                else:
                    st.error(f"Failed to launch {name}:\n{output}")
        with col2:
            if st.button(f"Stop {name}"):
                try:
                    container = client.containers.get(name.lower())
                    container.stop()
                    st.success(f"{name} container stopped.")
                except docker.errors.NotFound:
                    st.error(f"No running {name} container found.")
                except Exception as e:
                    st.error(f"Error stopping {name}: {e}")
        with col3:
            if st.button(f"Remove {name}"):
                try:
                    container = client.containers.get(name.lower())
                    container.remove(force=True)
                    st.success(f"{name} container removed.")
                except docker.errors.NotFound:
                    st.error(f"No {name} container found.")
                except Exception as e:
                    st.error(f"Error removing {name}: {e}")

elif menu == "Launch LR Model with Flask API":
    st.write("Launches a Flask API with a simple Linear Regression model inside Docker.")
    if st.button("Launch Flask App"):
        try:
            import os
            os.makedirs("lr_model", exist_ok=True)
            with open("lr_model/app.py", "w") as f:
                f.write("""from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
app = Flask(__name__)
model = LinearRegression()
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])
model.fit(X_train, y_train)
@app.route('/')
def home(): return 'LR Model Flask API'
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    pred = model.predict(np.array(data['input']).reshape(-1, 1)).tolist()
    return jsonify(prediction=pred)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)""")
            with open("lr_model/Dockerfile", "w") as f:
                f.write("""FROM python:3.9
WORKDIR /app
COPY app.py ./
RUN pip install flask scikit-learn numpy
CMD ["python", "app.py"]""")
            success, output = run_shell_command("docker build -t lr-flask-app ./lr_model")
            if success:
                st.success("Docker image built successfully.")
                success, output = run_shell_command("docker run -d -p 5000:5000 lr-flask-app")
                if success:
                    st.success("LR Flask app running on port 5000.")
                else:
                    st.error(f"Failed to start Flask container:\n{output}")
            else:
                st.error(f"Failed to build Docker image:\n{output}")
        except Exception as e:
            st.error(f"Error during Flask app setup: {e}")

elif menu == "Build from Dockerfile":
    path = st.text_input("Dockerfile Directory", "./")
    tag = st.text_input("Image Tag", "custom-image")
    if st.button("Build and Run"):
        success, output = run_shell_command(f"docker build -t {tag} {path}")
        if success:
            st.success(f"Docker image '{tag}' built successfully.")
            success, output = run_shell_command(f"docker run -dit {tag}")
            if success:
                st.success(f"Container started from image '{tag}'.")
            else:
                st.error(f"Failed to run container:\n{output}")
        else:
            st.error(f"Failed to build image:\n{output}")

elif menu == "Commit Container to Image":
    cid = st.text_input("Running Container ID or Name")
    name = st.text_input("New Image Name", "my-image")
    if st.button("Commit"):
        if not cid.strip() or not name.strip():
            st.error("Container ID/Name and new image name cannot be empty.")
        else:
            try:
                container = client.containers.get(cid.strip())
                image = container.commit(repository=name.strip())
                st.success(f"Committed container '{cid}' to image '{name}'.")
            except docker.errors.NotFound:
                st.error(f"No container found with ID/Name '{cid}'.")
            except Exception as e:
                st.error(f"Error committing container: {e}")

elif menu == "Docker-in-Docker (DinD)":
    if st.button("Launch DinD"):
        st.write("Checking for existing 'dind' container...")
        try:
            existing = client.containers.get("dind")
            existing.stop()
            existing.remove()
            st.info("Existing DinD container stopped and removed.")
        except docker.errors.NotFound:
            pass
        except Exception as e:
            st.error(f"Error cleaning up existing DinD container: {e}")
        success, output = run_shell_command("docker run --privileged --name dind -d docker:dind")
        if success:
            st.success("Docker-in-Docker container launched.")
        else:
            st.error(f"Failed to launch DinD container:\n{output}")

elif menu == "📘 Docker Knowledge Section":
    st.title("📘 Docker Knowledge Zone")

    with st.expander("🤔 What is Docker?"):
        st.markdown("""
Docker is a platform for building, running, and sharing containerized applications.
- **Created by:** Solomon Hykes (2013)
- **Purpose:** Package software in containers for consistency and portability.

Think of Docker like a magical box. Whatever software and configuration you put inside it—will run the same way on your friend's laptop, your server, or the cloud.

**Why it matters:** Without Docker, deploying applications can be like giving someone a dish with no recipe. Docker ensures the recipe is followed every time.
        """)

    with st.expander("📦 Docker Commit vs Dockerfile"):
        st.markdown("### 🔄 Docker Commit")
        st.markdown("- Creates a snapshot of a running container (like a quick photo of your setup). Useful for saving container state after manual config.")
        st.code("docker commit <container_id> image_name", language="bash")
        st.markdown("### 📿 Dockerfile")
        st.markdown("- Acts as a recipe file for building images in a consistent, versioned way.")
        st.code("""FROM python:3.9
COPY . /app
RUN pip install -r requirements.txt
CMD [\"python\", \"app.py\"]""", language="dockerfile")
        st.markdown("**Build with:**")
        st.code("docker build -t myimage .", language="bash")
        st.markdown("Use Dockerfile when you want automation, and commit when you're experimenting live.")

    with st.expander("🚪 How to Install Docker on Windows"):
        st.markdown("""
Installing Docker on Windows is easy with Docker Desktop. It creates a lightweight VM using WSL2 to run Linux containers.

### 💪 Steps:
1. Install WSL2: `wsl --install`
2. Download and install Docker Desktop: [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
3. Enable WSL2 integration in Docker settings.
4. Run `docker run hello-world` to verify.

**Extra Tip:** Use "Resources" settings in Docker Desktop to manage CPU, RAM and disk usage.
        """)

    with st.expander("🧪 Why Companies Use Docker - Case Study"):
        st.markdown("""
### 🏢 Real Use-Cases of Docker

**Netflix** – Uses Docker for microservices deployment and fast CI/CD.

**PayPal** – Reduced infrastructure complexity and build times.

**Spotify** – Better testing and deployment cycles with containerized services.

**Tesla** – Containers power internal tooling for faster releases.

### 🔍 What They Gain:
- Portability across environments
- Speed in testing and deployment
- Security through isolation
- Cleaner DevOps automation

**Bonus Insight:** Docker makes "fail fast, recover fast" possible. This agility is key in modern tech.
        """)

    with st.expander("🧠 How to Use 'systemctl' Inside Docker"):
        st.markdown("""
By default, `systemctl` doesn't work in Docker because containers lack a full init system.

### 🛠️ Fix:
Use a systemd-enabled base image (e.g. CentOS or Ubuntu with systemd).

```Dockerfile
FROM centos/systemd
RUN yum -y install httpd
CMD ["/usr/sbin/init"]
```
Run with:
```bash
docker run --privileged -v /sys/fs/cgroup:/sys/fs/cgroup:ro yourimage
```
This lets your container behave more like a virtual machine.

**Pro Tip:** Always test service-based apps with systemd in a privileged container.
        """)

    with st.expander("🌐 Display GUI Applications from Docker"):
        st.markdown("""
Docker can display GUI apps using X11 on Linux or tools like VcXsrv on Windows.

### 💻 Linux:
```bash
xhost +
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix x11-app
```

### 🌟 Windows (VcXsrv):
```cmd
set DISPLAY=host.docker.internal:0.0
docker run -e DISPLAY=%DISPLAY% gui-app
```
Great for running apps like Firefox, VSCode, or VLC from a container.

**Heads up:** You might need to allow public access in your X11/VcXsrv settings.
        """)

    with st.expander("🔊 Enable Audio in Docker Containers"):
        st.markdown("""
To play audio from containers (e.g. VLC), use PulseAudio.

### 🛠️ Linux Setup:
```bash
xhost +local:
docker run -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.config/pulse:/root/.config/pulse \
    --device /dev/snd vlc-image
```
This forwards your system's sound server to the container.

**Hint:** Check for audio group permissions if sound doesn't work.
        """)

    with st.expander("🐳 Docker-in-Docker (DinD) Explained"):
        st.markdown("""
Docker-in-Docker lets you run Docker **inside** a Docker container. Perfect for CI pipelines like GitLab CI.

### 🔧 Setup:
```bash
docker run --privileged --name dind -d docker:dind
```

### 🪨 Caution:
DinD is powerful but can be insecure. Avoid using in production environments unless isolated.

**Use Case:** Great for teaching, sandboxing, or automated container builds inside CI.
        """)

elif menu == "📺 Emotion Detection (Webcam)":
    st.subheader("📺 Real-Time Emotion Detection (face detection demo)")
    webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)

elif menu == "Exit":
    st.warning("Close the browser tab or stop the Streamlit server to exit.")
    # Voice feedback removed to fix asyncio loop error.