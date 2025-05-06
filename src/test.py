from azure.storage.blob import BlobServiceClient
connection_string = "DefaultEndpointsProtocol=https;AccountName=quartzocapital;AccountKey=cMgSYkaQlZ1HarEdqVAq1CQ34YHrDUQ7JzaeZByV2akuYg/MQecmXtu4u4y6Sl2bggnBMqfxnX+v+AStTfjoGA==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("landing-zone")
print(container_client.get_container_properties())