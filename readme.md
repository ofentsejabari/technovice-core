# Create a Django project

Create a Django starter project by building the image from the build context defined in the `Dockerfile`.

1. Change to the root of your project directory.
2. Create the Django project by running the `docker-compose run` command as follows.

  >  `sudo docker-compose run web django-admin startproject technoviceml . `

This instructs Compose to run `django-admin startproject technoviceml` in a container, using the `app-core` service’s image and configuration. 
Because the `app-core` image doesn’t exist yet, Compose builds it from the current directory, as specified by the `build: .` line in docker-compose.yml.

Once the web service image is built, Compose runs it and executes the `django-admin startproject` command in the container. This command instructs Django
to create a set of files and directories representing a Django project.

If you are running Docker on Linux, the files django-admin created are owned by root. This happens because the container runs as the root user.
Change the ownership of the new files.

`sudo chown -R $USER:$USER .`
