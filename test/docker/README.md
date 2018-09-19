# Docker image for hls4ml with Vivado

Put the Vivado installation folder here and provide the path to license server.
For example:

```
docker build --network=host -t hls4ml-with-vivado --build-arg LICENSE_SERVER="1234@myserver" .
```

By default, version 2017.2 of Vivado is used. Edit the `Dockerfile` to change the version used. 