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
                // sh 'ls -la'
            }
        }

        stage('Verificar archivos') {
            steps {
                sh 'ls -la'
            }
        }

        stage('Instalación de dependencias') {
            steps {
                sh '''
                pip install --no-cache-dir -r requirements.txt
                echo "Dependencias instaladas correctamente"
                '''
            }
        }

        stage('Pruebas básicas del dataset') {
            steps {
                sh '''
                python3 -c "
import pandas as pd
df = pd.read_csv('dataset/sdss_sample.csv')
assert df.shape[0] > 0, 'El dataset está vacío'
assert 'class' in df.columns, 'Falta la columna class'
assert 'redshift' in df.columns, 'Falta la columna redshift'
for col in ['u', 'g', 'r', 'i', 'z']:
    assert col in df.columns, f'Falta la columna {col}'
print(f'Dataset OK: {df.shape[0]} filas x {df.shape[1]} columnas')
print(f'Clases encontradas: {df[\"class\"].unique().tolist()}')
"
                '''
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

        stage('Ejecutar script principal') {
            steps {
                sh '''
                docker exec ml_container python3 src/main.py
                echo "Script principal ejecutado correctamente"
                '''
            }
        }

        stage('Ejecutar notebooks') {
            steps {
                sh '''
                docker exec ml_container jupyter nbconvert \
                    --to notebook \
                    --execute notebooks/data_analysis.ipynb \
                    --output outputs/data_analysis_output.ipynb

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
                  
                echo "Notebooks ejecutados correctamente"
                '''
            }
        }

        stage('Almacenamiento de artefactos') {
            steps {
                sh '''
                docker cp ml_container:/app/outputs ./outputs
                docker cp ml_container:/app/models ./models
                echo "Artefactos copiados correctamente"
                ls -la outputs/plots/
                ls -la outputs/metrics/
                ls -la models/
                '''
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
                archiveArtifacts artifacts: 'models/**/*.pkl', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado correctamente'
            echo 'Metricas  -> outputs/metrics/'
            echo 'Graficas  -> outputs/plots/'
            echo 'Modelos   -> models/'
        }
        failure {
            echo 'Error en el pipeline, revisar logs'
            sh 'docker rm -f ml_container || true'
        }
        always {
            sh 'docker rm -f ml_container || true'
        }
    }
}