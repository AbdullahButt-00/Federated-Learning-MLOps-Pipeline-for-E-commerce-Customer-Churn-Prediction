pipeline {
    agent any
    
    environment {
        // Docker registry (update if using Docker Hub or private registry)
        REGISTRY = 'localhost:5000'  // Change to your registry
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        
        // Kubernetes namespace
        K8S_NAMESPACE = 'churn-prediction'
        
        // MLflow
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
        
        // Dataset path - can be overridden by parameters
        DATASET_PATH = "${params.DATASET_PATH ?: 'E_Commerce_Dataset.xlsx'}"
    }
    
    parameters {
        string(
            name: 'DATASET_PATH',
            defaultValue: 'E_Commerce_Dataset.xlsx',
            description: 'Path to the dataset file (for drift testing, use drifted dataset)'
        )
        booleanParam(
            name: 'SKIP_TESTS',
            defaultValue: false,
            description: 'Skip testing phase'
        )
        booleanParam(
            name: 'FORCE_DEPLOY',
            defaultValue: false,
            description: 'Force deployment even if model accuracy is low'
        )
    }
    
    stages {
        stage('Cleanup') {
            steps {
                echo '🧹 Cleaning workspace...'
                deleteDir()
            }
        }
        
        stage('Checkout') {
            steps {
                echo '📦 Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup') {
            steps {
                echo '⚙️  Setting up environment...'
                sh '''
                    # Ensure required directories exist
                    mkdir -p preprocessed_data
                    mkdir -p federated_data
                    mkdir -p federated_data/round_evaluation
                    
                    # Check if dataset exists
                    if [ ! -f "${DATASET_PATH}" ]; then
                        echo "❌ Dataset not found: ${DATASET_PATH}"
                        exit 1
                    fi
                    
                    echo "✓ Using dataset: ${DATASET_PATH}"
                '''
            }
        }
        
        stage('Start MLflow Server') {
            steps {
                echo '🚀 Starting MLflow server...'
                sh '''
                    # Check if MLflow is already running
                    if pgrep -f "mlflow server" > /dev/null; then
                        echo "✓ MLflow server already running"
                    else
                        echo "Starting MLflow server..."
                        nohup mlflow server \
                            --host 0.0.0.0 \
                            --port 5000 \
                            --backend-store-uri sqlite:///mlflow.db \
                            --default-artifact-root ./mlflow \
                            > mlflow.log 2>&1 &
                        
                        # Wait for MLflow to be ready
                        echo "Waiting for MLflow to be ready..."
                        for i in {1..30}; do
                            if curl -s http://localhost:5000/health > /dev/null; then
                                echo "✓ MLflow server is ready"
                                break
                            fi
                            echo "Waiting... ($i/30)"
                            sleep 2
                        done
                    fi
                    
                    # Verify MLflow is accessible
                    curl -f http://localhost:5000/health || exit 1
                '''
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                echo '🔄 Running data preprocessing...'
                sh '''
                    python preprocess.py \
                        --dataset "${DATASET_PATH}" \
                        --output-folder preprocessed_data \
                        --clients 3
                    
                    # Verify preprocessed data exists
                    if [ ! -f "preprocessed_data/preprocessor.pkl" ]; then
                        echo "❌ Preprocessing failed: preprocessor.pkl not found"
                        exit 1
                    fi
                    
                    echo "✓ Preprocessing complete"
                '''
            }
        }
        
        stage('Model Training') {
            steps {
                echo '🤖 Training federated model...'
                sh '''
                    python training_MLFlow.py \
                        --dataset "${DATASET_PATH}" \
                        --data-folder preprocessed_data \
                        --output-folder federated_data \
                        --num-rounds 20 \
                        --batch-size 8 \
                        --test-frac 0.2
                    
                    # Verify model was created
                    if [ ! -f "federated_data/federated_churn_model.h5" ]; then
                        echo "❌ Training failed: model file not found"
                        exit 1
                    fi
                    
                    echo "✓ Training complete"
                '''
            }
        }
        
        stage('Extract Model Metrics') {
            steps {
                echo '📊 Extracting model metrics...'
                script {
                    // Read the training log or metrics file to get accuracy
                    def metricsFile = 'federated_data/round_evaluation/per_round_metrics.csv'
                    if (fileExists(metricsFile)) {
                        def metrics = readFile(metricsFile)
                        echo "Model Metrics:\n${metrics}"
                        
                        // Extract final accuracy (last line of CSV)
                        def lines = metrics.split('\n')
                        if (lines.size() > 1) {
                            def lastLine = lines[-1]
                            def values = lastLine.split(',')
                            if (values.size() > 4) {
                                env.MODEL_ACCURACY = values[4]  // eval_accuracy column
                                echo "✓ Final Model Accuracy: ${env.MODEL_ACCURACY}"
                                
                                // Check if accuracy meets threshold
                                def accuracy = env.MODEL_ACCURACY.toFloat()
                                if (accuracy < 0.80 && !params.FORCE_DEPLOY) {
                                    error("❌ Model accuracy (${accuracy}) is below 80% threshold. Deployment cancelled.")
                                }
                            }
                        }
                    } else {
                        echo "⚠️  Metrics file not found, skipping accuracy check"
                    }
                }
            }
        }
        
        stage('Build Docker Images') {
            steps {
                echo '🐳 Building Docker images...'
                sh '''
                    # Build preprocessing image
                    docker build -f Dockerfile.preprocess \
                        -t ${REGISTRY}/churn-preprocess:${IMAGE_TAG} \
                        -t ${REGISTRY}/churn-preprocess:latest .
                    
                    # Build training image
                    docker build -f Dockerfile.training \
                        -t ${REGISTRY}/churn-training:${IMAGE_TAG} \
                        -t ${REGISTRY}/churn-training:latest .
                    
                    # Build serving image
                    docker build -f Dockerfile.serving \
                        -t ${REGISTRY}/churn-serving:${IMAGE_TAG} \
                        -t ${REGISTRY}/churn-serving:latest .
                    
                    echo "✓ Docker images built successfully"
                '''
            }
        }
        
        stage('Push Docker Images') {
            steps {
                echo '📤 Pushing Docker images to registry...'
                sh '''
                    # Push images (skip if using local registry that doesn't require push)
                    # Uncomment if using remote registry:
                    # docker push ${REGISTRY}/churn-preprocess:${IMAGE_TAG}
                    # docker push ${REGISTRY}/churn-preprocess:latest
                    # docker push ${REGISTRY}/churn-training:${IMAGE_TAG}
                    # docker push ${REGISTRY}/churn-training:latest
                    # docker push ${REGISTRY}/churn-serving:${IMAGE_TAG}
                    # docker push ${REGISTRY}/churn-serving:latest
                    
                    echo "✓ Docker images ready for deployment"
                '''
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                echo '☸️  Deploying to Kubernetes...'
                sh '''
                    # Apply namespace
                    kubectl apply -f k8s/namespace.yaml
                    
                    # Apply PVCs
                    kubectl apply -f k8s/pvc.yaml
                    
                    # Deploy MLflow
                    kubectl apply -f k8s/mlflow-deployment.yaml
                    
                    # Wait for MLflow to be ready
                    kubectl wait --for=condition=available --timeout=120s \
                        deployment/mlflow -n ${K8S_NAMESPACE} || true
                    
                    # Deploy Prometheus & Grafana
                    kubectl apply -f k8s/prometheus-rbac.yaml
                    kubectl apply -f k8s/prometheus-configmap.yaml
                    kubectl apply -f k8s/prometheus-deployment.yaml
                    kubectl apply -f k8s/grafana-datasource-configmap.yaml
                    kubectl apply -f k8s/grafana-dashboard-config.yaml
                    kubectl apply -f k8s/grafana-dashboard-json.yaml
                    kubectl apply -f k8s/grafana-deployment.yaml
                    
                    # Deploy API
                    kubectl apply -f k8s/api-deployment.yaml
                    
                    # Wait for API deployment
                    kubectl wait --for=condition=available --timeout=180s \
                        deployment/churn-api -n ${K8S_NAMESPACE}
                    
                    echo "✓ Deployment complete"
                    
                    # Show deployment status
                    kubectl get pods -n ${K8S_NAMESPACE}
                    kubectl get services -n ${K8S_NAMESPACE}
                '''
            }
        }
        
        stage('Start Docker Compose Services') {
            steps {
                echo '🐳 Starting Docker Compose services...'
                sh '''
                    # Stop existing containers
                    docker-compose down || true
                    
                    # Start services
                    docker-compose up -d
                    
                    # Wait for services to be healthy
                    echo "Waiting for services to be ready..."
                    sleep 10
                    
                    # Check service health
                    docker-compose ps
                    
                    # Verify API is responding
                    for i in {1..30}; do
                        if curl -s http://localhost:8000/health > /dev/null; then
                            echo "✓ Churn API is ready"
                            break
                        fi
                        echo "Waiting for API... ($i/30)"
                        sleep 2
                    done
                    
                    echo "✓ Docker Compose services started"
                '''
            }
        }
        
        stage('Health Check') {
            steps {
                echo '🏥 Running health checks...'
                sh '''
                    # Check API health
                    curl -f http://localhost:8000/health || exit 1
                    echo "✓ API health check passed"
                    
                    # Check model info
                    curl -f http://localhost:8000/model-info || exit 1
                    echo "✓ Model info endpoint working"
                    
                    # Check Prometheus
                    curl -f http://localhost:9090/-/healthy || exit 1
                    echo "✓ Prometheus health check passed"
                    
                    # Check Grafana
                    curl -f http://localhost:3000/api/health || exit 1
                    echo "✓ Grafana health check passed"
                    
                    echo "✓ All health checks passed"
                '''
            }
        }
        
        stage('Reload Model in API') {
            steps {
                echo '🔄 Reloading model in API...'
                sh '''
                    # Trigger model reload to pick up new Production version
                    curl -X POST http://localhost:8000/reload-model || true
                    
                    # Wait a bit for reload
                    sleep 5
                    
                    # Verify new model is loaded
                    curl -s http://localhost:8000/model-info | grep -q "current_version" && \
                        echo "✓ Model reloaded successfully" || \
                        echo "⚠️  Could not verify model reload"
                '''
            }
        }
    }
    
    post {
        success {
            echo '''
            ============================================================
            ✓ PIPELINE COMPLETED SUCCESSFULLY
            ============================================================
            
            Services are now running:
            - Churn API: http://localhost:8000
            - MLflow: http://localhost:5000
            - Prometheus: http://localhost:9090
            - Grafana: http://localhost:3000 (admin/root)
            
            API Documentation: http://localhost:8000/docs
            Model Metrics: http://localhost:8000/metrics
            
            ============================================================
            '''
        }
        
        failure {
            echo '''
            ============================================================
            ❌ PIPELINE FAILED
            ============================================================
            
            Check the logs above for error details.
            Common issues:
            - Dataset not found
            - Model accuracy below threshold
            - Docker build failures
            - Kubernetes deployment issues
            
            ============================================================
            '''
        }
        
        always {
            echo '📋 Archiving artifacts...'
            archiveArtifacts artifacts: '''
                federated_data/**/*.png,
                federated_data/**/*.svg,
                federated_data/**/*.csv,
                federated_data/**/*.html,
                federated_data/**/*.h5,
                preprocessed_data/metadata.json
            ''', allowEmptyArchive: true
            
            // Clean up old Docker images
            sh '''
                docker image prune -f || true
            '''
        }
    }
}