pipeline {
    agent any

    stages {

        stage('Limpiar workspace') {
            steps {
                deleteDir()
            }
        }

        stage('Clonar repositorio') {
            steps {
                git branch: 'teo', url: 'https://github.com/Foster-11/ML-Astronomic.git'
            }
        }

        stage('Verificar archivos') {
            steps {
                sh 'ls -la'
            }
        }

        stage('Test Docker') {
            steps {
                sh 'docker version'
                sh 'docker ps'
            }
        }

        stage('Construir imagen Docker') {
            steps {
                sh 'docker build -t proyecto_ml .'
            }
        }

        stage('Ejecutar contenedor') {
            steps {
                sh '''
                docker rm -f ml_container || true
                docker run -d -p 8888:8888 --name ml_container proyecto_ml
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado correctamente 🚀'
        }
        failure {
            echo 'Error en el pipeline ❌ revisar logs'
        }
    }
}