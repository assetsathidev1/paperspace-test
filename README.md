# paperspace-test
Repo to test out paperspace platforms


## Paperspace Notes: 
- Video+ Tutorial slightly older but still works. 
- All deployments come with `/healthcheck`
- Models to be Uploaded and path to be specified as `/opt/models/<model_file_name>`
- TF, Onnx models supported out of box; also has custom option

## Paperspace Questions during setup:
- Where did we provide the metadata for the [model](https://youtu.be/voyqmlYOIH0?feature=shared&t=266)

- Check the image below this is slightly confusing. 
![confusing ux](document-resources/deploy_model_ux.png)

- New UI changes breaks the old demo.
    - `<endpoint>/v1/models/fashion-mnist/metadata` does not seem to work, tried my model name, model id and deployment name as well.
    - Tried a `CLI` deployment with `cli-fashion-deply-spec.yaml` worked as expected on metadata and prediction
    - Updated the `path:"/opt/mosqqhgydnbf3wu"` to `path:"/opt/models/fashion-mnist"` in `paperspace.json` to `modif_config_paperspace.json` and uploaded this new config in deployment and everything works again. 

- Will disabling the deployment stop the billing? (i doubt it)


## Using HF-FastAPI-Paperspace Template
- Base image is large, taking a lot of time to build locally
