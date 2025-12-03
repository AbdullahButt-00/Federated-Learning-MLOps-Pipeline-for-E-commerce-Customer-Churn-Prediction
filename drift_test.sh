#!/bin/bash

# Test script to generate predictions with data drift
# This creates normal data first, then introduces drift

API_URL="http://192.168.49.2:32176/predict"

echo "=========================================="
echo "Starting Drift Detection Test"
echo "=========================================="

# Arrays for random data generation
DEVICES=("Mobile Phone" "Computer")
PAYMENT_MODES=("Credit Card" "Debit Card" "Cash on Delivery" "UPI" "E wallet")
GENDERS=("Male" "Female")
CATEGORIES=("Laptop & Accessory" "Mobile Phone" "Fashion" "Grocery" "Others")
MARITAL_STATUS=("Single" "Married" "Divorced")

# Function to generate random number in range
random_range() {
    local min=$1
    local max=$2
    echo $(awk -v min=$min -v max=$max 'BEGIN{srand(); print min+rand()*(max-min)}')
}

# Function to generate random integer in range
random_int() {
    local min=$1
    local max=$2
    echo $(shuf -i $min-$max -n 1)
}

# Function to pick random element from array
random_element() {
    local arr=("$@")
    local size=${#arr[@]}
    local idx=$(random_int 0 $((size-1)))
    echo "${arr[$idx]}"
}

echo ""
echo "Phase 1: Generating 100 NORMAL predictions..."
echo "This establishes the reference distribution"
echo ""

for i in {1..100}; do
    # Normal ranges - typical customer behavior
    TENURE=$(random_range 1 60)
    CITY_TIER=$(random_int 1 3)
    WAREHOUSE_DIST=$(random_range 5 30)
    DEVICE=$(random_element "${DEVICES[@]}")
    PAYMENT=$(random_element "${PAYMENT_MODES[@]}")
    GENDER=$(random_element "${GENDERS[@]}")
    HOURS=$(random_range 1 5)
    NUM_DEVICES=$(random_int 1 5)
    CATEGORY=$(random_element "${CATEGORIES[@]}")
    SATISFACTION=$(random_int 1 5)
    MARITAL=$(random_element "${MARITAL_STATUS[@]}")
    NUM_ADDR=$(random_int 1 5)
    COMPLAIN=$(random_int 0 1)
    ORDER_HIKE=$(random_range 10 30)
    COUPON=$(random_range 0 5)
    ORDER_COUNT=$(random_range 1 10)
    DAYS_SINCE=$(random_range 0 10)
    CASHBACK=$(random_range 50 300)

    curl -s -X POST $API_URL \
        -H "Content-Type: application/json" \
        -d "{
            \"Tenure\": $TENURE,
            \"PreferredLoginDevice\": \"$DEVICE\",
            \"CityTier\": $CITY_TIER,
            \"WarehouseToHome\": $WAREHOUSE_DIST,
            \"PreferredPaymentMode\": \"$PAYMENT\",
            \"Gender\": \"$GENDER\",
            \"HourSpendOnApp\": $HOURS,
            \"NumberOfDeviceRegistered\": $NUM_DEVICES,
            \"PreferedOrderCat\": \"$CATEGORY\",
            \"SatisfactionScore\": $SATISFACTION,
            \"MaritalStatus\": \"$MARITAL\",
            \"NumberOfAddress\": $NUM_ADDR,
            \"Complain\": $COMPLAIN,
            \"OrderAmountHikeFromlastYear\": $ORDER_HIKE,
            \"CouponUsed\": $COUPON,
            \"OrderCount\": $ORDER_COUNT,
            \"DaySinceLastOrder\": $DAYS_SINCE,
            \"CashbackAmount\": $CASHBACK
        }" > /dev/null &
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ✓ Sent $i/100 normal predictions"
    fi
    
    sleep 0.05
done

wait
echo ""
echo "Phase 1 Complete! Reference data established."
echo "Waiting 5 seconds before introducing drift..."
sleep 5

echo ""
echo "=========================================="
echo "Phase 2: Generating 60 DRIFTED predictions..."
echo "Introducing significant distribution changes"
echo "=========================================="
echo ""

for i in {1..60}; do
    # DRIFTED ranges - abnormal customer behavior
    # Much longer tenure (older customers)
    TENURE=$(random_range 40 100)
    
    # Higher tier cities only
    CITY_TIER=$(random_int 1 1)
    
    # Much longer distances (relocation pattern)
    WAREHOUSE_DIST=$(random_range 40 80)
    
    # Preferring computers over mobile (behavior shift)
    DEVICE="Computer"
    
    PAYMENT=$(random_element "${PAYMENT_MODES[@]}")
    GENDER=$(random_element "${GENDERS[@]}")
    
    # Much more app usage (increased engagement)
    HOURS=$(random_range 6 12)
    
    # More devices registered (tech-savvy)
    NUM_DEVICES=$(random_int 5 10)
    
    CATEGORY=$(random_element "${CATEGORIES[@]}")
    
    # Lower satisfaction (declining service quality)
    SATISFACTION=$(random_int 1 2)
    
    MARITAL=$(random_element "${MARITAL_STATUS[@]}")
    
    # More addresses (frequent movers)
    NUM_ADDR=$(random_int 6 15)
    
    # More complaints (service degradation)
    COMPLAIN=1
    
    # Huge order amount increases (inflation/price hikes)
    ORDER_HIKE=$(random_range 50 100)
    
    # Heavy coupon usage (price-sensitive)
    COUPON=$(random_range 10 20)
    
    # Many more orders (power users)
    ORDER_COUNT=$(random_range 15 30)
    
    # Much longer gaps between orders (reduced frequency)
    DAYS_SINCE=$(random_range 15 40)
    
    # Much higher cashback (promotional period)
    CASHBACK=$(random_range 500 1000)

    curl -s -X POST $API_URL \
        -H "Content-Type: application/json" \
        -d "{
            \"Tenure\": $TENURE,
            \"PreferredLoginDevice\": \"$DEVICE\",
            \"CityTier\": $CITY_TIER,
            \"WarehouseToHome\": $WAREHOUSE_DIST,
            \"PreferredPaymentMode\": \"$PAYMENT\",
            \"Gender\": \"$GENDER\",
            \"HourSpendOnApp\": $HOURS,
            \"NumberOfDeviceRegistered\": $NUM_DEVICES,
            \"PreferedOrderCat\": \"$CATEGORY\",
            \"SatisfactionScore\": $SATISFACTION,
            \"MaritalStatus\": \"$MARITAL\",
            \"NumberOfAddress\": $NUM_ADDR,
            \"Complain\": $COMPLAIN,
            \"OrderAmountHikeFromlastYear\": $ORDER_HIKE,
            \"CouponUsed\": $COUPON,
            \"OrderCount\": $ORDER_COUNT,
            \"DaySinceLastOrder\": $DAYS_SINCE,
            \"CashbackAmount\": $CASHBACK
        }" > /dev/null &
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ⚠️  Sent $i/60 DRIFTED predictions"
    fi
    
    sleep 0.05
done

wait

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ 100 normal predictions (reference data)"
echo "  ⚠️  60 drifted predictions (monitoring data)"
echo ""
echo "Expected Drift in Features:"
echo "  - Tenure (much higher: 40-100 vs 1-60)"
echo "  - WarehouseToHome (much higher: 40-80 vs 5-30)"
echo "  - HourSpendOnApp (much higher: 6-12 vs 1-5)"
echo "  - NumberOfDeviceRegistered (higher: 5-10 vs 1-5)"
echo "  - SatisfactionScore (lower: 1-2 vs 1-5)"
echo "  - NumberOfAddress (much higher: 6-15 vs 1-5)"
echo "  - OrderAmountHikeFromlastYear (much higher: 50-100 vs 10-30)"
echo "  - CouponUsed (much higher: 10-20 vs 0-5)"
echo "  - OrderCount (much higher: 15-30 vs 1-10)"
echo "  - DaySinceLastOrder (much higher: 15-40 vs 0-10)"
echo "  - CashbackAmount (much higher: 500-1000 vs 50-300)"
echo ""
echo "Check your Grafana dashboard for drift alerts!"
echo "Dashboard: http://localhost:3000"
echo ""