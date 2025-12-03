#!/bin/bash

# Traffic generator for populating Grafana dashboard

NAMESPACE="churn-prediction"

echo "=================================================="
echo "🚦 Traffic Generator"
echo "=================================================="
echo ""

# Get API URL
API_URL=$(minikube service churn-api-service -n $NAMESPACE --url 2>/dev/null)

if [ -z "$API_URL" ]; then
    echo "❌ Could not get API URL"
    echo "Run: minikube service churn-api-service -n $NAMESPACE --url"
    exit 1
fi

echo "API URL: $API_URL"
echo ""

# Test payloads with variety
PAYLOADS=(
    '{"Tenure": 12.0, "PreferredLoginDevice": "Mobile Phone", "CityTier": 1, "WarehouseToHome": 15.0, "PreferredPaymentMode": "Credit Card", "Gender": "Male", "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 3, "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 5, "MaritalStatus": "Single", "NumberOfAddress": 2, "Complain": 0, "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0, "OrderCount": 5.0, "DaySinceLastOrder": 3.0, "CashbackAmount": 150.0}'
    '{"Tenure": 24.0, "PreferredLoginDevice": "Computer", "CityTier": 2, "WarehouseToHome": 20.0, "PreferredPaymentMode": "Debit Card", "Gender": "Female", "HourSpendOnApp": 2.0, "NumberOfDeviceRegistered": 2, "PreferedOrderCat": "Fashion", "SatisfactionScore": 3, "MaritalStatus": "Married", "NumberOfAddress": 3, "Complain": 1, "OrderAmountHikeFromlastYear": 10.0, "CouponUsed": 2.0, "OrderCount": 3.0, "DaySinceLastOrder": 5.0, "CashbackAmount": 100.0}'
    '{"Tenure": 6.0, "PreferredLoginDevice": "Mobile Phone", "CityTier": 3, "WarehouseToHome": 25.0, "PreferredPaymentMode": "UPI", "Gender": "Male", "HourSpendOnApp": 1.0, "NumberOfDeviceRegistered": 1, "PreferedOrderCat": "Mobile", "SatisfactionScore": 2, "MaritalStatus": "Single", "NumberOfAddress": 1, "Complain": 1, "OrderAmountHikeFromlastYear": 5.0, "CouponUsed": 0.0, "OrderCount": 2.0, "DaySinceLastOrder": 10.0, "CashbackAmount": 50.0}'
    '{"Tenure": 36.0, "PreferredLoginDevice": "Computer", "CityTier": 1, "WarehouseToHome": 10.0, "PreferredPaymentMode": "Credit Card", "Gender": "Female", "HourSpendOnApp": 4.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Grocery", "SatisfactionScore": 5, "MaritalStatus": "Divorced", "NumberOfAddress": 2, "Complain": 0, "OrderAmountHikeFromlastYear": 20.0, "CouponUsed": 3.0, "OrderCount": 8.0, "DaySinceLastOrder": 2.0, "CashbackAmount": 200.0}'
)

# Function to make a prediction
make_prediction() {
    local payload=$1
    local result=$(curl -s -w "\n%{http_code}" -X POST $API_URL/predict \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)
    
    local http_code=$(echo "$result" | tail -1)
    local response=$(echo "$result" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        echo "✓"
        return 0
    else
        echo "✗ ($http_code)"
        return 1
    fi
}

# Generate continuous traffic
echo "Generating traffic... (Press Ctrl+C to stop)"
echo ""
echo "Mode: Continuous (1 request every 2 seconds)"
echo ""

COUNT=0
SUCCESS=0
FAIL=0

while true; do
    # Select random payload
    RANDOM_INDEX=$((RANDOM % ${#PAYLOADS[@]}))
    PAYLOAD="${PAYLOADS[$RANDOM_INDEX]}"
    
    # Make prediction
    ((COUNT++))
    printf "[%03d] " $COUNT
    
    if make_prediction "$PAYLOAD"; then
        ((SUCCESS++))
    else
        ((FAIL++))
    fi
    
    # Show stats every 10 requests
    if [ $((COUNT % 10)) -eq 0 ]; then
        echo ""
        echo "--- Stats: $SUCCESS success, $FAIL failed, $COUNT total ---"
        echo ""
        
        # Show current metrics
        METRICS=$(curl -s $API_URL/metrics 2>/dev/null | grep "predictions_total\|prediction_latency_seconds_count" | grep -v "#")
        if [ ! -z "$METRICS" ]; then
            echo "Current metrics:"
            echo "$METRICS"
            echo ""
        fi
    fi
    
    # Wait before next request
    sleep 2
done