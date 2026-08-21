# 🐳 Docker CLI Cheatsheet

## Inspect and debug

```bash
docker ps                     # running containers
docker ps -a                  # all containers (including stopped)
docker images                 # list images
docker image ls               # same as above
docker images -a              # include intermediate images

docker system df              # disk usage summary
docker system df -v           # detailed disk usage

docker stats                  # live CPU / memory usage
docker inspect <id>           # detailed JSON info
docker logs <container>       # view logs
```

---

## 🧹 Cleanup (VERY IMPORTANT)

```bash
docker container prune        # remove stopped containers
docker image prune            # remove dangling images
docker image prune -a         # remove ALL unused images
docker volume prune           # remove unused volumes
docker network prune          # remove unused networks
```

### 🔥 Full cleanup
```bash
docker system prune           # safe cleanup
docker system prune -a        # remove all unused images too
docker system prune -a --volumes  # ⚠️ also deletes volumes
```

---

## 💣 Force delete everything

```bash
docker images -q | xargs docker rmi -f
```

Safer version (no duplicate errors):
```bash
docker images -q | sort -u | xargs docker rmi -f
```

---

## 📦 Containers

```bash
docker run <image>            # run container
docker run -it <image> bash   # interactive shell

docker start <container>
docker stop <container>
docker restart <container>

docker rm <container>         # remove container
docker rm -f <container>      # force remove
```

Remove all stopped containers:
```bash
docker ps -aq | xargs docker rm
```

---

## 🧱 Images

```bash
docker pull <image>           # download image
docker build -t name:tag .    # build image
docker rmi <image>            # remove image
docker rmi -f <image>         # force remove
```

Filter:
```bash
docker images -f dangling=true
```

---

## 📊 Sorting / Finding large images

```bash
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | sort -h
```

---

## 💾 Volumes

```bash
docker volume ls
docker volume inspect <name>
docker volume rm <name>
docker volume prune           # remove unused volumes
```

Check disk usage:
```bash
du -sh /var/lib/docker/volumes/*
```

---

## ⚙️ Build & Performance

```bash
DOCKER_BUILDKIT=1 docker build .
```

Clean build cache:
```bash
docker builder prune
docker builder prune -a
```

---

## 🧠 Resource Control

```bash
docker run --cpus="0.5" <image>
docker run -m 512m <image>
```

---

## 🔍 Useful Patterns

### Remove specific group of images
```bash
docker images "tb__*" -q | sort -u | xargs docker rmi
```

### Check container sizes
```bash
docker ps -a --size
```

### Find large logs
```bash
du -sh /var/lib/docker/containers/*
```

---

## ⚠️ Key Concepts

- **Image** = template (read-only layers)
- **Container** = image + writable layer
- **Stopped container still uses disk**
- **Volumes persist data even after container removal**
- **Images can share layers (not additive size)**

---

## 🎯 Golden Commands (most useful)

```bash
docker system df
docker system prune -a --volumes
docker stats
docker images
docker ps -a
```

---

## 🚀 Pro Tips

- Use `prune` regularly to avoid disk bloat
- Watch out for **volumes** (can silently eat TBs)
- Avoid mixing `docker` and `sudo docker`
- Use `sort -u` when deleting many images
- Disk I/O is often the real bottleneck in builds