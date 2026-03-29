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

        stage('Ejecutar notebooks') {
            steps {
                sh '''
                docker exec ml_container jupyter nbconvert \
                    --to notebook \
                    --execute notebooks/knn.ipynb \
                    --output outputs/knn_output.ipynb

                docker exec ml_container jupyter nbconvert \
                    --to notebook \
                    --execute notebooks/linearRegression.ipynb \
                    --output outputs/linearRegression_output.ipynb

                docker exec ml_container jupyter nbconvert \
                    --to notebook \
                    --execute notebooks/kmeans.ipynb \
                    --output outputs/kmeans_output.ipynb
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