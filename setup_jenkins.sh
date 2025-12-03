#!/bin/bash

echo "============================================================"
echo "Jenkins Pipeline Setup for Churn Prediction"
echo "============================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Jenkins is running
echo -e "\n${YELLOW}Checking Jenkins status...${NC}"
if systemctl is-active --quiet jenkins; then
    echo -e "${GREEN}✓ Jenkins is running${NC}"
else
    echo -e "${RED}❌ Jenkins is not running${NC}"
    echo "Starting Jenkins..."
    sudo systemctl start jenkins
    sleep 10
fi

# Get Jenkins URL
JENKINS_URL="http://localhost:8090"
echo -e "\n${YELLOW}Jenkins URL: ${JENKINS_URL}${NC}"

# Check if Jenkins is accessible
if curl -s "${JENKINS_URL}" > /dev/null; then
    echo -e "${GREEN}✓ Jenkins is accessible${NC}"
else
    echo -e "${RED}❌ Jenkins is not accessible${NC}"
    exit 1
fi

echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}MANUAL SETUP REQUIRED IN JENKINS WEB UI${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "\n1. Open Jenkins: ${GREEN}${JENKINS_URL}${NC}"
echo -e "   Login with your credentials"

echo -e "\n2. Install Required Plugins:"
echo -e "   - Go to: Manage Jenkins > Manage Plugins"
echo -e "   - Install: ${GREEN}Pipeline${NC}, ${GREEN}Git${NC}, ${GREEN}Docker Pipeline${NC}, ${GREEN}Kubernetes${NC}"

echo -e "\n3. Create New Pipeline Job:"
echo -e "   - Click: ${GREEN}New Item${NC}"
echo -e "   - Name: ${GREEN}churn-prediction-pipeline${NC}"
echo -e "   - Type: ${GREEN}Pipeline${NC}"
echo -e "   - Click OK"

echo -e "\n4. Configure Pipeline:"
echo -e "   - In Pipeline section, select: ${GREEN}Pipeline script from SCM${NC}"
echo -e "   - SCM: ${GREEN}Git${NC}"
echo -e "   - Repository URL: ${GREEN}[Your repo URL]${NC}"
echo -e "   - Script Path: ${GREEN}Jenkinsfile${NC}"
echo -e "   - OR: Select ${GREEN}Pipeline script${NC} and paste Jenkinsfile content"

echo -e "\n5. Add Parameters:"
echo -e "   - Check: ${GREEN}This project is parameterized${NC}"
echo -e "   - Add String Parameter:"
echo -e "     Name: ${GREEN}DATASET_PATH${NC}"
echo -e "     Default: ${GREEN}E_Commerce_Dataset.xlsx${NC}"
echo -e "   - Add Boolean Parameters:"
echo -e "     ${GREEN}SKIP_TESTS${NC} (default: false)"
echo -e "     ${GREEN}FORCE_DEPLOY${NC} (default: false)"

echo -e "\n6. Get Jenkins API Token:"
echo -e "   - Click your username (top right) > ${GREEN}Configure${NC}"
echo -e "   - Under API Token, click ${GREEN}Add new Token${NC}"
echo -e "   - Name it: ${GREEN}alertmanager-webhook${NC}"
echo -e "   - Copy the token and save it"

echo -e "\n7. Set Environment Variables:"
echo -e "   ${GREEN}export JENKINS_TOKEN='your-token-here'${NC}"
echo -e "   ${GREEN}export JENKINS_USER='your-username'${NC}"

echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}GENERATE DRIFT DATASET${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "\nGenerate synthetic drift dataset:"
echo -e "${GREEN}python generate_drift_dataset.py --drift-type moderate${NC}"

echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}START JENKINS TRIGGER SERVICE${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "\nStart the webhook receiver:"
echo -e "${GREEN}export JENKINS_TOKEN='your-token'${NC}"
echo -e "${GREEN}export JENKINS_USER='your-username'${NC}"
echo -e "${GREEN}python jenkins_trigger.py${NC}"

echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}UPDATE ALERTMANAGER CONFIG${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "\nUpdate alertmanager.yml webhook URL to:"
echo -e "${GREEN}http://localhost:5001/trigger-retraining${NC}"

echo -e "\nRestart Alertmanager:"
echo -e "${GREEN}docker-compose restart alertmanager${NC}"

echo -e "\n${YELLOW}============================================================${NC}"
echo -e "${YELLOW}TEST THE PIPELINE${NC}"
echo -e "${YELLOW}============================================================${NC}"

echo -e "\n1. Manual Test:"
echo -e "   - Go to Jenkins job"
echo -e "   - Click: ${GREEN}Build with Parameters${NC}"
echo -e "   - Use default parameters"
echo -e "   - Click: ${GREEN}Build${NC}"

echo -e "\n2. Drift Test:"
echo -e "   - Generate drift dataset"
echo -e "   - Send predictions to API to trigger drift detection"
echo -e "   - Alertmanager will trigger retraining automatically"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup instructions complete!${NC}"
echo -e "${GREEN}============================================================${NC}"