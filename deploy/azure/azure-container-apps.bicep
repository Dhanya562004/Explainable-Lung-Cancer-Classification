@description('Azure location for resources')
param location string = resourceGroup().location

@description('Prefix for resource names')
param namePrefix string = 'lungcancer'

@description('Container image tag')
param imageTag string = 'v1.0.0'

var acrName = '${namePrefix}acr${uniqueString(resourceGroup().id)}'
var envName = '${namePrefix}-env'
var apiAppName = '${namePrefix}-api'
var uiAppName = '${namePrefix}-ui'

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource managedEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {}
}

resource apiContainerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: apiAppName
  location: location
  properties: {
    managedEnvironmentId: managedEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      secrets: [
        {
          name: 'registry-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
      registries: [
        {
          server: acr.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'registry-password'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'fastapi-api'
          image: '${acr.loginServer}/explainable-lung-cancer-api:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }
          env: [
            {
              name: 'ENVIRONMENT'
              value: 'production'
            },
            {
              name: 'MAX_UPLOAD_SIZE_MB'
              value: '10'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health/live'
                port: 8000
              }
              initialDelaySeconds: 15
              periodSeconds: 10
            },
            {
              type: 'Readiness'
              httpGet: {
                path: '/health/ready'
                port: 8000
              }
              initialDelaySeconds: 15
              periodSeconds: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

output apiFqdn string = apiContainerApp.properties.configuration.ingress.fqdn
output acrLoginServer string = acr.loginServer
