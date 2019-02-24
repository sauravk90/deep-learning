from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()  # class instantiation
arguments = {"keywords":"Bike","limit":2000, "print_urls":True, "chromedriver":"chromedriver.exe"}
paths = response.download(arguments)
print(paths)   #printing absolute paths of the downloaded images
