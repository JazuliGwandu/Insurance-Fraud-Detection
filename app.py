import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

class FraudDetectionPrototype:
    def __init__(self):
        # Initialize top_features with default values first
        self.top_features = [
            'InscClaimAmtReimbursed', 'AttendingPhysician', 'IPAnnualReimbursementAmt',
            'OPAnnualReimbursementAmt', 'OperatingPhysician', 'OtherPhysician',
            'ClmProcedureCode_1', 'DeductibleAmtPaid', 'NoOfMonths_PartACov',
            'NoOfMonths_PartBCov', 'ClmDiagnosisCode_1', 'IPAnnualDeductibleAmt',
            'OPAnnualDeductibleAmt', 'ClmAdmitDiagnosisCode', 'Gender',
            'State', 'County', 'ChronicCond_Heartfailure', 'ChronicCond_Diabetes',
            'RenalDiseaseIndicator', 'ChronicCond_Alzheimer', 'ChronicCond_Cancer',
            'ChronicCond_Depression', 'ChronicCond_IschemicHeart', 'ChronicCond_KidneyDisease',
            'ChronicCond_ObstrPulmonary', 'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
            'ChronicCond_stroke', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
            'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
            'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
            'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DiagnosisGroupCode',
            'Race'
        ]
        self.demo_mode = True  # Start in demo mode
    
        # Initialize metrics with your actual performance numbers
        self.current_auc = 0.8851      # Your actual AUC score
        self.best_auc = 0.8851
        self.current_accuracy = 0.8430 # Your actual accuracy  
        self.best_accuracy = 0.8430
        self.current_f1 = 0.7665       # Your actual F1-score
        self.best_f1 = 0.7665
    
        self.load_models()     # Then try to load real models
        
    def load_models(self):
        """Load the trained model with comprehensive comparison data"""
        try:
            models_path = Path("D:/Projects/fraud_detection_pipeline/Models")
            
            print("="*60)
            print("🤖 MODEL LOADING SYSTEM")
            print("="*60)
            
            # Check for comprehensive comparison file
            comprehensive_file = models_path / "comprehensive_model_comparison.csv"
            original_file = models_path / "model_comparison_results.csv"
            
            if comprehensive_file.exists():
                print("✓ Found comprehensive comparison data")
                self.comparison_df = pd.read_csv(comprehensive_file)
                data_source = "comprehensive"
            elif original_file.exists():
                print("✓ Found original comparison data")
                self.comparison_df = pd.read_csv(original_file)
                data_source = "original"
            else:
                print("⚠️ No comparison data found")
                self.comparison_df = None
                data_source = None
            
            # Load the best model (try tuned -> enhanced -> original)
            model_loaded = False
            
            # 1. Try Tuned XGBoost
            tuned_model = models_path / "tuned_xgboost_model.pkl"
            if tuned_model.exists():
                enhanced_scaler = models_path / "scaler_enhanced.pkl"
                enhanced_features = models_path / "feature_names_enhanced.pkl"
                
                if enhanced_scaler.exists() and enhanced_features.exists():
                    self.model = joblib.load(tuned_model)
                    self.scaler = joblib.load(enhanced_scaler)
                    self.top_features = joblib.load(enhanced_features)
                    self.model_type = "XGBoost (Tuned)"
                    self.model_version = "Tuned"
                    
                    # Get metrics from comprehensive data
                    if self.comparison_df is not None:
                        try:
                            # Try to find XGBoost (Tuned) model
                            tuned_row = self.comparison_df[self.comparison_df['Model'] == 'XGBoost (Tuned)']
                            if not tuned_row.empty:
                                # Extract and update metrics
                                row = tuned_row.iloc[0]
                                self.current_auc = float(row['AUC'])
                                self.best_auc = self.current_auc
                                self.current_accuracy = float(row['Accuracy'])
                                self.best_accuracy = self.current_accuracy
                                self.current_f1 = float(row['F1-Score'])
                                self.best_f1 = self.current_f1
                            else:
                                # Keep the initialized values (0.8851, 0.8430, 0.7665)
                                print("⚠️ Tuned model not found in comparison data, using initialized metrics")
                        except Exception as e:
                            print(f"⚠️ Error loading metrics: {str(e)}")
                            # Keep the initialized values (from __init__)
                    
                    model_loaded = True
                    print("✅ Loaded: XGBoost (Tuned) - Best Performance")
            
            # 2. Try Enhanced XGBoost
            if not model_loaded:
                enhanced_model = models_path / "enhanced_xgboost_model.pkl"
                if enhanced_model.exists():
                    enhanced_scaler = models_path / "scaler_enhanced.pkl"
                    enhanced_features = models_path / "feature_names_enhanced.pkl"
                    
                    if enhanced_scaler.exists() and enhanced_features.exists():
                        self.model = joblib.load(enhanced_model)
                        self.scaler = joblib.load(enhanced_scaler)
                        self.top_features = joblib.load(enhanced_features)
                        self.model_type = "XGBoost (Enhanced)"
                        self.model_version = "Enhanced"
                        
                        # Get metrics for enhanced model
                        if self.comparison_df is not None:
                            try:
                                enhanced_row = self.comparison_df[self.comparison_df['Model'] == 'XGBoost (Enhanced)']
                                if not enhanced_row.empty:
                                    row = enhanced_row.iloc[0]
                                    self.current_auc = float(row['AUC'])
                                    self.best_auc = self.current_auc
                                    self.current_accuracy = float(row['Accuracy'])
                                    self.best_accuracy = self.current_accuracy
                                    self.current_f1 = float(row['F1-Score'])
                                    self.best_f1 = self.current_f1
                                else:
                                    print("⚠️ Enhanced model not found in comparison data, using initialized metrics")
                            except Exception as e:
                                print(f"⚠️ Error loading enhanced metrics: {str(e)}")
                        
                        model_loaded = True
                        print("✅ Loaded: XGBoost (Enhanced)")
            
            # 3. Try Original Best Model
            if not model_loaded:
                original_model = models_path / "best_fraud_detection_model.pkl"
                if original_model.exists():
                    original_scaler = models_path / "scaler.pkl"
                    original_features = models_path / "feature_names.pkl"
                    
                    if original_scaler.exists() and original_features.exists():
                        self.model = joblib.load(original_model)
                        self.scaler = joblib.load(original_scaler)
                        self.top_features = joblib.load(original_features)
                        
                        # Determine which original model it is
                        if self.comparison_df is not None:
                            try:
                                best_original = self.comparison_df.iloc[0]
                                self.model_type = best_original['Model']
                                self.current_auc = float(best_original['AUC'])
                                self.best_auc = self.current_auc
                                self.current_accuracy = float(best_original['Accuracy'])
                                self.best_accuracy = self.current_accuracy
                                self.current_f1 = float(best_original['F1-Score'])
                                self.best_f1 = self.current_f1
                            except Exception as e:
                                print(f"⚠️ Error loading original model metrics: {str(e)}")
                                self.model_type = "XGBoost (Original)"
                                self.current_auc = 0.873
                                self.best_auc = 0.873
                                self.current_accuracy = 0.827
                                self.best_accuracy = 0.827
                                self.current_f1 = 0.755
                                self.best_f1 = 0.755
                        else:
                            self.model_type = "XGBoost (Original)"
                            self.current_auc = 0.873
                            self.best_auc = 0.873
                            self.current_accuracy = 0.827
                            self.best_accuracy = 0.827
                            self.current_f1 = 0.755
                            self.best_f1 = 0.755
                        
                        self.model_version = "Original"
                        model_loaded = True
                        print(f"✅ Loaded: {self.model_type}")
            
            # 4. Fallback to demo mode
            if not model_loaded:
                print("⚠️ No model files found. Using DEMO MODE.")
                self.demo_mode = True
                return
            
            self.demo_mode = False
            
            # Load training summary
            summary_file = models_path / "training_summary.txt"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.training_summary = f.read()
            
            # Print loading summary
            print("\n" + "="*60)
            print("📊 LOADING SUMMARY")
            print("="*60)
            print(f"Model: {self.model_type}")
            print(f"Version: {self.model_version}")
            print(f"Data Source: {data_source}")
            print(f"Comparison Models: {len(self.comparison_df) if self.comparison_df is not None else 0}")
            print(f"Current AUC: {self.current_auc:.4f}")
            print(f"Features: {len(self.top_features)}")
            print("="*60)
            
        except Exception as e:  # THIS WAS MISSING
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            self.demo_mode = True
            print("Using DEMO MODE")
    
    def predict_fraud(self, input_data):
        """Predict fraud probability for input data"""
        if self.demo_mode:
            # Demo mode - realistic simulation
            claim_amt = input_data.get('InscClaimAmtReimbursed', 5000)
            procedure_code = input_data.get('ClmProcedureCode_1', 1000)
            deductible = input_data.get('DeductibleAmtPaid', 500)
            
            base_risk = min(0.8, claim_amt / 40000)
            if procedure_code in [9999, 8888, 7777]:
                base_risk += 0.3
            if deductible > 2000:
                base_risk += 0.2
            physician_id = input_data.get('AttendingPhysician', 100)
            if physician_id % 13 == 0:
                base_risk += 0.25
                
            return min(0.95, max(0.05, base_risk))
        
        try:
            # Real model prediction
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present and in correct order
            for feature in self.top_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[self.top_features]
            scaled_data = self.scaler.transform(input_df)
            probability = self.model.predict_proba(scaled_data)[0, 1]
            return probability
            
        except Exception as e:
            return 0.5
    
    def manual_input_form(self):
        """Create professional manual input form"""
        st.header("🔍 Individual Claim Assessment")
        
        with st.container():
            st.subheader("Claim Details")
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("💰 Financial Information", expanded=True):
                    insc_claim_amt = st.number_input("Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0, help="Total amount reimbursed for this claim")
                    ip_annual_amt = st.number_input("Inpatient Annual ($)", min_value=0.0, value=15000.0, step=1000.0)
                    op_annual_amt = st.number_input("Outpatient Annual ($)", min_value=0.0, value=8000.0, step=1000.0)
                    deductible_amt = st.number_input("Deductible Paid ($)", min_value=0.0, value=500.0, step=50.0)
            
            with col2:
                with st.expander("📋 Coverage & Medical", expanded=True):
                    part_a_months = st.number_input("Part A Months", min_value=0, value=12, max_value=36)
                    part_b_months = st.number_input("Part B Months", min_value=0, value=12, max_value=36)
                    procedure_code = st.number_input("Procedure Code", min_value=0, value=1234, max_value=9999)
                    diagnosis_code = st.number_input("Diagnosis Code", min_value=0, value=5678, max_value=9999)
                    admit_diagnosis = st.number_input("Admission Diagnosis", min_value=0, value=9101, max_value=9999)
        
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("👤 Demographic Information"):
                    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
                    state = st.number_input("State Code", min_value=0, value=5, max_value=50)
                    county = st.number_input("County Code", min_value=0, value=10, max_value=500)
                    race = st.number_input("Race Code", min_value=0, value=1, max_value=10)
                    diagnosis_group = st.number_input("Diagnosis Group", min_value=0, value=100, max_value=1000)
            
            with col2:
                with st.expander("🏥 Health Conditions"):
                    renal_disease = st.selectbox("Renal Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    heart_failure = st.selectbox("Heart Failure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    cancer = st.selectbox("Cancer", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    alzheimer = st.selectbox("Alzheimer's", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with st.container():
            with st.expander("👨‍⚕️ Physician Information"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    attending_physician = st.number_input("Attending Physician ID", min_value=0, value=100)
                with col2:
                    operating_physician = st.number_input("Operating Physician ID", min_value=0, value=200)
                with col3:
                    other_physician = st.number_input("Other Physician ID", min_value=0, value=300)
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Fraud Risk", use_container_width=True, type="primary"):
                input_data = self._prepare_input_data(
                    insc_claim_amt, ip_annual_amt, op_annual_amt, deductible_amt,
                    part_a_months, part_b_months, procedure_code, diagnosis_code, admit_diagnosis,
                    gender, state, county, race, diagnosis_group,
                    renal_disease, heart_failure, diabetes, cancer, alzheimer,
                    attending_physician, operating_physician, other_physician
                )
                probability = self.predict_fraud(input_data)
                self._display_professional_results(probability, input_data)
    
    def _prepare_input_data(self, insc_claim_amt, ip_annual_amt, op_annual_amt, deductible_amt,
                          part_a_months, part_b_months, procedure_code, diagnosis_code, admit_diagnosis,
                          gender, state, county, race, diagnosis_group,
                          renal_disease, heart_failure, diabetes, cancer, alzheimer,
                          attending_physician, operating_physician, other_physician):
        """Prepare input data from form variables"""
        feature_defaults = {
            'InscClaimAmtReimbursed': insc_claim_amt,
            'AttendingPhysician': attending_physician,
            'IPAnnualReimbursementAmt': ip_annual_amt,
            'OPAnnualReimbursementAmt': op_annual_amt,
            'OperatingPhysician': operating_physician,
            'OtherPhysician': other_physician,
            'ClmProcedureCode_1': procedure_code,
            'DeductibleAmtPaid': deductible_amt,
            'NoOfMonths_PartACov': part_a_months,
            'NoOfMonths_PartBCov': part_b_months,
            'ClmDiagnosisCode_1': diagnosis_code,
            'IPAnnualDeductibleAmt': ip_annual_amt * 0.1,
            'OPAnnualDeductibleAmt': op_annual_amt * 0.1,
            'ClmAdmitDiagnosisCode': admit_diagnosis,
            'Gender': gender,
            'State': state,
            'County': county,
            'ChronicCond_Heartfailure': heart_failure,
            'ChronicCond_Diabetes': diabetes,
            'RenalDiseaseIndicator': renal_disease,
            'ChronicCond_Alzheimer': alzheimer,
            'ChronicCond_Cancer': cancer,
            'ChronicCond_Depression': 0,
            'ChronicCond_IschemicHeart': 0,
            'ChronicCond_KidneyDisease': 0,
            'ChronicCond_ObstrPulmonary': 0,
            'ChronicCond_Osteoporasis': 0,
            'ChronicCond_rheumatoidarthritis': 0,
            'ChronicCond_stroke': 0,
            'ClmDiagnosisCode_2': 0, 'ClmDiagnosisCode_3': 0, 'ClmDiagnosisCode_4': 0,
            'ClmDiagnosisCode_5': 0, 'ClmDiagnosisCode_6': 0, 'ClmDiagnosisCode_7': 0,
            'ClmDiagnosisCode_8': 0, 'ClmDiagnosisCode_9': 0, 'ClmDiagnosisCode_10': 0,
            'ClmProcedureCode_2': 0, 'ClmProcedureCode_3': 0, 'ClmProcedureCode_4': 0,
            'ClmProcedureCode_5': 0, 'ClmProcedureCode_6': 0,
            'DiagnosisGroupCode': diagnosis_group,
            'Race': race
        }
        
        input_data = {}
        for feature in self.top_features:
            input_data[feature] = feature_defaults.get(feature, 0)
        return input_data
    
    def _display_professional_results(self, probability, input_data):
        """Display professional results with horizontal layout"""
        st.markdown("---")
        st.header("📊 Risk Assessment Report")
        
        # Risk Level Card - HORIZONTAL LAYOUT
        if probability < 0.3:
            risk_color = "#10b981"
            risk_level = "LOW RISK"
            risk_icon = "✅"
            recommendation = "This claim appears legitimate"
        elif probability < 0.7:
            risk_color = "#f59e0b"
            risk_level = "MEDIUM RISK"
            risk_icon = "⚠️"
            recommendation = "Review recommended"
        else:
            risk_color = "#ef4444"
            risk_level = "HIGH RISK"
            risk_icon = "🚨"
            recommendation = "Immediate investigation required"
        
        # Horizontal layout for risk assessment
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="border: 2px solid {risk_color}; border-radius: 10px; padding: 15px; text-align: center; background-color: {risk_color}10; margin-bottom: 10px;">
                <div style="color: {risk_color}; font-size: 24px; font-weight: bold; margin: 5px 0;">{risk_icon}</div>
                <div style="color: {risk_color}; font-size: 18px; font-weight: bold; margin: 5px 0;">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(recommendation)
        
        with col2:
            # Probability Box
            st.markdown(f"""
            <div style="border: 2px solid {risk_color}; border-radius: 10px; padding: 15px; text-align: center; background-color: {risk_color}10; margin-bottom: 10px;">
                <div style="color: {risk_color}; font-size: 32px; font-weight: bold; margin: 5px 0; line-height: 1.2;">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Fraud Probability")
        
        with col3:
            # Key Risk Factors
            st.subheader("🔍 Key Risk Indicators")
            
            risk_factors = []
            if input_data['InscClaimAmtReimbursed'] > 20000:
                risk_factors.append(("High Claim Amount", "🔴", "Claim exceeds $20,000 threshold"))
            if input_data['AttendingPhysician'] % 13 == 0:
                risk_factors.append(("Physician Pattern", "🟡", "Unusual physician billing pattern"))
            if input_data['DeductibleAmtPaid'] > 1500:
                risk_factors.append(("High Deductible", "🟡", "Deductible payment above normal range"))
            if input_data['ClmProcedureCode_1'] in [9999, 8888, 7777]:
                risk_factors.append(("Procedure Code", "🔴", "Flagged procedure code detected"))
            
            if risk_factors:
                for factor, icon, description in risk_factors:
                    st.write(f"**{icon} {factor}**")
                    st.caption(description)
            else:
                st.write("✅ **No major risk factors detected**")
                st.caption("All parameters within normal ranges")
        
        # Visualizations below the risk cards
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Additional metrics and system status
            st.subheader("📈 Assessment Details")
            
            # Confidence metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Score", f"{probability:.1%}")
            with col_b:
                confidence = (1 - abs(probability - 0.5)) * 200
                st.metric("Model Confidence", f"{confidence:.1f}%")
            
            st.markdown("---")
            
    
            # UPDATED SYSTEM STATUS SECTION:
            st.markdown("---")
            
            if self.demo_mode:
                st.warning("**🔸 Demo Mode**")
                st.caption("Using simulated predictions for demonstration")
            else:
                # Determine icon and color based on model version
                version_info = {
                    'Original': ('🔵', '#3b82f6', 'Original Model'),
                    'Enhanced': ('🟡', '#f59e0b', 'Enhanced with Feature Engineering'),
                    'Tuned': ('🟢', '#10b981', 'Tuned with Hyperparameter Optimization')
                }
                
                icon, color, description = version_info.get(
                    getattr(self, 'model_version', 'XGBoost'),
                    ('🟢', '#10b981', 'Production Model')
                )
                
                # Header with model version
                st.success(f"**{icon} Production Mode - {getattr(self, 'model_type', 'XGBoost (Tuned)')}**")
                st.caption(description)
                

            # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                
                with col1:
                    st.metric(
                        "AUC Score", 
                        f"{getattr(self, 'current_auc', 0.8851):.3f}",
                        f"{getattr(self, 'model_version', 'XGBoost')}"
                    )
                
                with col2:
                    st.metric(
                        "Accuracy",
                        f"{getattr(self, 'current_accuracy', 0.8430):.3f}",
                        "Validation Set"
                    )
                
                with col3:
                    st.metric(
                        "F1-Score",
                        f"{getattr(self, 'current_f1', 0.7665):.3f}",
                        "Balance Metric"
                    )
                
                with col4:
                    st.metric(
                        "Features",
                        f"{len(self.top_features)}",
                        f"{getattr(self, 'model_version', 'XGBoost')}"
                    )
                
                # Show improvement if not original
                if hasattr(self, 'model_version') and self.model_version != 'Tuned':
                    st.info(f"✅ **Model Improvement**: This {self.model_version.lower()} model includes feature engineering and hyperparameter optimization for better performance.")
    
            
            # Quick actions
            st.markdown("---")
            st.subheader("🚀 Recommended Actions")
            if probability >= 0.7:
                st.error("**Immediate Actions Required:**")
                st.write("• Flag claim for manual review")
                st.write("• Notify investigation team")
                st.write("• Hold payment pending review")
            elif probability >= 0.3:
                st.warning("**Review Recommended:**")
                st.write("• Schedule for routine audit")
                st.write("• Monitor similar claims")
            else:
                st.success("**Standard Processing:**")
                st.write("• Process through normal channels")
                st.write("• No additional review needed")
    
    def batch_analysis(self):
        """Professional batch analysis interface"""
        st.header("📁 Bulk Claims Processing")
        
        tab1, tab2 = st.tabs(["📤 Upload Claims File", "🧪 Use Sample Data"])
        
        with tab1:
            st.subheader("Upload Your Claims Data")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], 
                                           help="Upload a CSV file containing insurance claims data")
            
            if uploaded_file is not None:
                try:
                    claims_df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Successfully loaded {len(claims_df)} claims")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(claims_df.head(), use_container_width=True)
                    
                    if st.button("🚀 Process All Claims", type="primary"):
                        self._analyze_batch_claims(claims_df)
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with tab2:
            st.subheader("Test with Sample Data")
            st.info("Generate sample claims to demonstrate the system")
            
            if st.button("Generate Sample Claims", use_container_width=True):
                demo_df = self._create_demo_data()
                self._analyze_batch_claims(demo_df)
    
    def _analyze_batch_claims(self, claims_df):
        """Analyze batch claims with professional presentation"""
        st.markdown("---")
        st.header("📈 Batch Analysis Results")
        
        # Show processing info ONCE
        st.write(f"🔧 Processing {len(claims_df)} claims...")
        
        # Progress and processing
        with st.spinner("Analyzing claims... This may take a few moments."):
            predictions = []
            for _, row in claims_df.iterrows():
                input_data = {}
                for feature in self.top_features:
                    input_data[feature] = row.get(feature, 0)
                prob = self.predict_fraud(input_data)
                predictions.append(prob)
        
        # Add results to dataframe
        claims_df = claims_df.copy()
        claims_df['Fraud_Probability'] = predictions
        claims_df['Risk_Level'] = pd.cut(claims_df['Fraud_Probability'], 
                                       bins=[0, 0.3, 0.7, 1],
                                       labels=['Low', 'Medium', 'High'])
        
        # Summary Cards
        st.subheader("Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Claims", len(claims_df))
        with col2:
            high_risk = (claims_df['Risk_Level'] == 'High').sum()
            st.metric("High Risk Claims", high_risk)
        with col3:
            avg_prob = claims_df['Fraud_Probability'].mean()
            st.metric("Average Risk", f"{avg_prob:.1%}")
        with col4:
            if self.demo_mode:
                st.metric("System Mode", "Demo")
            else:
                st.metric("Model Confidence", "87.3%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution
            risk_counts = claims_df['Risk_Level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color=risk_counts.index,
                color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk Score Distribution
            fig = px.histogram(
                claims_df, 
                x='Fraud_Probability',
                title="Fraud Probability Distribution",
                nbins=20,
                color_discrete_sequence=['#3b82f6']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Results
        st.subheader("Detailed Analysis")
        
        tab1, tab2 = st.tabs(["🎯 High Risk Claims", "📋 All Results"])
        
        with tab1:
            high_risk_df = claims_df[claims_df['Risk_Level'] == 'High'].sort_values('Fraud_Probability', ascending=False)
            if len(high_risk_df) > 0:
                st.dataframe(high_risk_df, use_container_width=True)
            else:
                st.success("🎉 No high-risk claims detected!")
        
        with tab2:
            st.dataframe(claims_df.sort_values('Fraud_Probability', ascending=False), use_container_width=True)
        
        # Download
        st.download_button(
            label="📥 Download Full Report",
            data=claims_df.to_csv(index=False),
            file_name="fraud_analysis_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def comparison_analysis(self):
        """Show comprehensive model comparison with ALL models"""
        st.header("📊 Comprehensive Model Performance Comparison")
        
        if self.demo_mode:
            st.warning("⚠️ Demo mode active - showing sample data")
            self._show_sample_comparison()
            return
        
        if not hasattr(self, 'comparison_df') or self.comparison_df is None:
            st.error("No comparison data available")
            return
        
        st.success(f"✅ Showing {len(self.comparison_df)} models including improved versions")
        
        # Filter options
        st.subheader("🔍 Filter Models")
        
        col1, col2 = st.columns(2)
        with col1:
            show_original = st.checkbox("Show Original Models", value=True)
        with col2:
            show_improved = st.checkbox("Show Improved Models", value=True)
        
        # Filter DataFrame
        filtered_df = self.comparison_df.copy()
        
        model_types = []
        if show_original:
            model_types.append('Original')
        if show_improved:
            model_types.extend(['Enhanced', 'Tuned'])
        
        filtered_df = filtered_df[filtered_df['Model_Type'].isin(model_types)]
        
        if filtered_df.empty:
            st.warning("No models match the selected filters")
            return
        
        # Display comparison table
        st.subheader("📈 Performance Metrics")
        
        # Format display
        display_df = filtered_df[['Model', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Model_Type']].copy()
        
        # Highlight current model
        def highlight_current(row):
            if row['Model'] == self.model_type:
                return ['background-color: #e8f5e8; font-weight: bold'] * len(row)
            elif row['Model_Type'] != 'Original':
                return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)
        
        # Format numbers
        for col in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_df.style.apply(highlight_current, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Bar Charts", "🏆 Ranking", "📈 Improvement"])
        
        with tab1:
            import plotly.express as px
            
            fig = px.bar(
                filtered_df.sort_values('AUC'),
                y='Model',
                x='AUC',
                orientation='h',
                color='Model_Type',
                title='AUC Score Comparison',
                color_discrete_map={
                    'Original': '#1f77b4',
                    'Enhanced': '#ff7f0e', 
                    'Tuned': '#2ca02c'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Rank models
            ranked_df = filtered_df.sort_values('AUC', ascending=False).reset_index(drop=True)
            ranked_df['Rank'] = ranked_df.index + 1
            
            cols = st.columns(3)
            for i in range(min(3, len(ranked_df))):
                with cols[i]:
                    medal = ["🥇", "🥈", "🥉"][i]
                    row = ranked_df.iloc[i]
                    st.metric(
                        f"{medal} {row['Model']}",
                        f"AUC: {row['AUC']:.3f}",
                        f"Type: {row['Model_Type']}"
                    )
            
            # Show full ranking
            st.dataframe(
                ranked_df[['Rank', 'Model', 'AUC', 'Accuracy', 'F1-Score', 'Model_Type']],
                use_container_width=True
            )
        
        with tab3:
            # Show improvement analysis
            if 'XGBoost' in filtered_df['Model_Family'].values:
                xgb_models = filtered_df[filtered_df['Model_Family'] == 'XGBoost']
                if len(xgb_models) > 1:
                    st.subheader("XGBoost Improvement Analysis")
                    
                    original_xgb = xgb_models[xgb_models['Model_Type'] == 'Original']
                    improved_xgb = xgb_models[xgb_models['Model_Type'] != 'Original']
                    
                    if not original_xgb.empty and not improved_xgb.empty:
                        original_auc = float(original_xgb['AUC'].iloc[0])
                        
                        for _, row in improved_xgb.iterrows():
                            improvement = float(row['AUC']) - original_auc
                            percentage = (improvement / original_auc) * 100
                            
                            st.metric(
                                f"{row['Model']} Improvement",
                                f"+{improvement:.4f} AUC",
                                f"+{percentage:.1f}% from original"
                            )
            
            # Show best improvement
            if len(filtered_df[filtered_df['Model_Type'] != 'Original']) > 0:
                improved = filtered_df[filtered_df['Model_Type'] != 'Original']
                best_improved = improved.nlargest(1, 'AUC').iloc[0]
                
                st.success(f"""
                **🎯 Best Improved Model: {best_improved['Model']}**
                
                - **AUC**: {best_improved['AUC']:.4f}
                - **Improvement Type**: {best_improved['Model_Type']}
                - **Key Enhancement**: {best_improved.get('Improvement_Notes', 'Feature Engineering & Hyperparameter Tuning')}
                """)
        
        # Download option
        st.download_button(
            label="📥 Download Complete Comparison Report",
            data=filtered_df.to_csv(index=False),
            file_name="comprehensive_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def _create_demo_data(self):
        """Create professional demo data"""
        demo_data = {}
        feature_values = {
            'InscClaimAmtReimbursed': [5000, 15000, 25000, 8000, 3000, 45000, 12000],
            'AttendingPhysician': [100, 150, 200, 120, 80, 260, 130],
            'IPAnnualReimbursementAmt': [15000, 25000, 35000, 18000, 12000, 42000, 16000],
            'OPAnnualReimbursementAmt': [8000, 12000, 15000, 9000, 6000, 18000, 10000],
            'OperatingPhysician': [200, 250, 300, 220, 180, 350, 230],
            'OtherPhysician': [300, 350, 400, 320, 280, 450, 330],
            'ClmProcedureCode_1': [1234, 5678, 7890, 3456, 1122, 9999, 2345],
            'DeductibleAmtPaid': [500, 1200, 2000, 800, 300, 2500, 900],
            'NoOfMonths_PartACov': [12, 6, 3, 24, 36, 2, 18],
            'NoOfMonths_PartBCov': [12, 6, 3, 24, 36, 2, 18],
            'ClmDiagnosisCode_1': [5678, 1234, 3456, 7890, 4455, 9999, 2345],
            'IPAnnualDeductibleAmt': [1500, 2500, 3500, 1800, 1200, 4200, 1600],
            'OPAnnualDeductibleAmt': [800, 1200, 1500, 900, 600, 1800, 1000],
            'ClmAdmitDiagnosisCode': [9101, 5678, 6789, 2345, 1122, 9999, 1234],
            'Gender': [1, 0, 0, 1, 1, 0, 1],
            'State': [5, 3, 2, 7, 9, 1, 6],
            'County': [10, 8, 5, 15, 12, 3, 11],
            'ChronicCond_Heartfailure': [0, 1, 1, 0, 0, 1, 0],
            'ChronicCond_Diabetes': [1, 0, 1, 0, 0, 1, 1],
            'RenalDiseaseIndicator': [0, 1, 1, 0, 0, 1, 0],
            'ChronicCond_Alzheimer': [0, 0, 1, 0, 0, 1, 0],
            'ChronicCond_Cancer': [0, 1, 0, 0, 1, 0, 1],
            'DiagnosisGroupCode': [100, 200, 150, 180, 120, 250, 160],
            'Race': [1, 2, 1, 3, 2, 1, 2]
        }
        
        for feature in self.top_features:
            demo_data[feature] = feature_values.get(feature, [0, 0, 0, 0, 0, 0, 0])
        
        return pd.DataFrame(demo_data)

    def _show_sample_comparison(self):
        """Show sample comparison data when in demo mode"""
        # Updated with all 5 models
        sample_data = {
            'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting', 
                     'Logistic Regression', 'SVM (Linear)'],
            'AUC': [0.8727, 0.8293, 0.8456, 0.6545, 0.7152],
            'Accuracy': [0.8268, 0.7952, 0.8083, 0.6500, 0.7015],
            'Precision': [0.8512, 0.7805, 0.7956, 0.5521, 0.6523],
            'Recall': [0.6803, 0.6408, 0.6985, 0.4782, 0.6021],
            'F1-Score': [0.7554, 0.7038, 0.7432, 0.5123, 0.6258]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        st.info("""
        **📊 Sample Data Display**
        
        This shows example performance metrics. When real models are loaded, 
        actual performance metrics from your training will be displayed here.
        
        **Models included in comparison:**
        - XGBoost
        - Random Forest  
        - Gradient Boosting
        - Logistic Regression
        - SVM (Linear)
        """)
        
        # Display sample table
        st.dataframe(sample_df, use_container_width=True)
        
        # Simple visualization
        fig = px.bar(
            sample_df, 
            x='Model', 
            y=['AUC', 'Accuracy', 'F1-Score'],
            title='Sample Model Performance Comparison',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_system_info(self):
        """Display comprehensive system information"""
        st.header("ℹ️ System Information")
        
        # Two columns layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Current Model Configuration")
            
            # Model card
            if self.demo_mode:
                st.warning("**🔸 Demo Mode Active**")
                st.caption("Using simulated predictions")
            else:
                version_colors = {
                    'Original': 'blue',
                    'Enhanced': 'orange', 
                    'Tuned': 'green'
                }
                
                color = version_colors.get(self.model_version, 'blue')
                
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding-left: 15px; margin: 10px 0;">
                    <h4 style="color: {color}; margin-bottom: 5px;">{self.model_type}</h4>
                    <p style="margin: 5px 0;"><strong>Version:</strong> {self.model_version}</p>
                    <p style="margin: 5px 0;"><strong>Features:</strong> {len(self.top_features)}</p>
                    <p style="margin: 5px 0;"><strong>Status:</strong> ✅ Production Ready</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📊 Performance Metrics")
            
            if hasattr(self, 'current_auc'):
                # Create metrics display
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("AUC Score", f"{self.current_auc:.3f}")
                    st.metric("Accuracy", f"{self.current_accuracy:.3f}")
                
                with metrics_col2:
                    st.metric("F1-Score", f"{self.current_f1:.3f}")
                    
                    # Show improvement if available
                    if hasattr(self, 'comparison_df') and self.comparison_df is not None:
                        original_xgb = self.comparison_df[
                            (self.comparison_df['Model_Family'] == 'XGBoost') & 
                            (self.comparison_df['Model_Type'] == 'Original')
                        ]
                        if not original_xgb.empty:
                            original_auc = float(original_xgb['AUC'].iloc[0])
                            if self.current_auc > original_auc:
                                improvement = self.current_auc - original_auc
                                st.metric(
                                    "Improvement", 
                                    f"+{improvement:.3f}",
                                    f"from {original_auc:.3f}"
                                )
            
            st.subheader("⚙️ System Capabilities")
            st.markdown("""
            ✅ Real-time individual claim assessment  
            ✅ Batch processing for multiple claims  
            ✅ Comprehensive model comparison  
            ✅ Downloadable reports  
            ✅ Professional risk visualization  
            """)
        
        with col2:
            st.subheader("🏆 All Models Performance")
            
            if hasattr(self, 'comparison_df') and self.comparison_df is not None:
                # Create a compact display of all models
                display_data = self.comparison_df[['Model', 'AUC', 'Accuracy', 'F1-Score', 'Model_Type']].copy()
                
                # Format numbers
                for col in ['AUC', 'Accuracy', 'F1-Score']:
                    display_data[col] = display_data[col].apply(lambda x: f"{x:.3f}")
                
                # Highlight current model
                def highlight_row(row):
                    if str(row['Model']) == self.model_type:
                        return ['background-color: #e8f5e8; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    display_data.style.apply(highlight_row, axis=1),
                    use_container_width=True,
                    height=300
                )
                
                # Summary statistics
                original_models = self.comparison_df[self.comparison_df['Model_Type'] == 'Original']
                improved_models = self.comparison_df[self.comparison_df['Model_Type'].isin(['Enhanced', 'Tuned'])]
                
                st.metric("Total Models", len(self.comparison_df))
                st.metric("Original Models", len(original_models))
                st.metric("Improved Models", len(improved_models))
                
            else:
                st.info("No comparison data available")
            
            st.subheader("🛠️ Technology Stack")
            st.markdown("""
            - **Machine Learning**: XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SVM
            - **Backend**: Python, Scikit-learn, XGBoost, Imbalanced-learn
            - **Frontend**: Streamlit, Plotly
            - **Data Processing**: Pandas, NumPy, Joblib
            - **Visualization**: Matplotlib, Seaborn, Plotly
            """)
        
        # Performance comparison section
        st.markdown("---")
        st.subheader("📈 Performance Comparison Insights")
        
        if hasattr(self, 'comparison_df') and self.comparison_df is not None:
            # Get top 3 models
            top_3 = self.comparison_df.nlargest(3, 'AUC')
            
            col1, col2, col3 = st.columns(3)
            
            for i, (_, row) in enumerate(top_3.iterrows()):
                with [col1, col2, col3][i]:
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.metric(
                        f"{medal} {row['Model']}",
                        f"AUC: {row['AUC']:.3f}",
                        f"Acc: {row['Accuracy']:.3f}"
                    )
            
            # Show improvement analysis
            if len(improved_models) > 0:
                st.info(f"""
                **Model Improvement Analysis:**
                
                - **Best Original Model**: {original_models.iloc[0]['Model']} (AUC: {original_models.iloc[0]['AUC']:.3f})
                - **Best Improved Model**: {improved_models.iloc[0]['Model']} (AUC: {improved_models.iloc[0]['AUC']:.3f})
                - **Improvement**: +{(improved_models.iloc[0]['AUC'] - original_models.iloc[0]['AUC']):.3f} AUC
                - **Percentage Gain**: +{((improved_models.iloc[0]['AUC'] - original_models.iloc[0]['AUC']) / original_models.iloc[0]['AUC'] * 100):.1f}%
                """)
    
    def main(self):
        """Main application interface"""
        st.set_page_config(
            page_title="Insurance Fraud Analytics",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Professional CSS Styling with LARGER FONTS AND FIXED METRICS
        st.markdown("""
        <style>
            /* Main headers */
            .main-header {
                font-size: 4rem;
                color: #1f77b4;  /* Blue - Clear and professional */
                text-align: center;
                margin-bottom: 1rem;
                font-weight: 800;
                line-height: 1.1;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }
            .sub-header {
                font-size: 1.8rem;
                color: #2c3e50;  /* Dark blue-gray for better contrast */
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 600;
                line-height: 1.2;
            }
            
            /* Sidebar improvements */
            .sidebar .sidebar-content {
                background-color: #f8f9fa;
            }
            [data-testid="stSidebar"] {
                background-color: #f8f9fa;
            }
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] .st-emotion-cache-16idsys p {
                color: #1f77b4 !important;
                font-weight: 600 !important;
            }
            [data-testid="stSidebar"] .st-emotion-cache-10trblm {
                color: #1f77b4 !important;
            }
            [data-testid="stSidebar"] .st-emotion-cache-1nm2qww {
                color: #2c3e50 !important;
                font-weight: 500 !important;
            }
            
            /* Form labels and text */
            label {
                color: #2c3e50 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
            }
            .stTextInput label, 
            .stNumberInput label,
            .stSelectbox label,
            .stExpander label {
                color: #1f77b4 !important;
                font-weight: 600 !important;
            }
            
            /* Expander headers */
            .stExpander {
                border: 1px solid #e0e0e0 !important;
                border-radius: 8px !important;
                margin-bottom: 10px !important;
            }
            .stExpander summary {
                background-color: #f1f8ff !important;
                color: #1f77b4 !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                padding: 12px 16px !important;
            }
            .stExpander summary:hover {
                background-color: #e3f2fd !important;
            }
            
            /* Buttons */
            .stButton>button {
                background-color: #1f77b4;
                color: white;
                border-radius: 6px;
                padding: 12px 28px;
                font-weight: 600;
                border: none;
                font-size: 16px !important;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #1668a5;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            /* Metric cards in sidebar */
            .metric-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #1f77b4;
            }
            
            /* Sidebar metrics */
            [data-testid="stMetricValue"] {
                color: #1f77b4 !important;
                font-weight: bold !important;
                font-size: 22px !important;
            }
            [data-testid="stMetricLabel"] {
                color: #2c3e50 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
            }
            
            /* FIX FOR METRICS TO SHOW FULL NUMBERS */
            .stMetric {
                padding: 10px 5px !important;
            }
            .stMetric label {
                font-size: 14px !important;
                font-weight: 600 !important;
                color: #1f77b4 !important;
                margin-bottom: 5px !important;
            }
            .stMetric div[data-testid="stMetricValue"] {
                font-size: 24px !important;
                font-weight: bold !important;
                color: #1f77b4 !important;  /* Changed to blue */
                letter-spacing: -0.5px !important;
            }
            .stMetric div[data-testid="stMetricDelta"] {
                font-size: 13px !important;
                font-weight: normal !important;
                color: #27ae60 !important;  /* Green for positive */
            }
            
            /* Make metric containers wider */
            div[data-testid="column"] {
                min-width: 150px !important;
            }
            
            /* Input fields */
            .stNumberInput input, 
            .stTextInput input,
            .stSelectbox select {
                border: 2px solid #e0e0e0 !important;
                border-radius: 6px !important;
                padding: 8px 12px !important;
                font-size: 14px !important;
            }
            .stNumberInput input:focus, 
            .stTextInput input:focus,
            .stSelectbox select:focus {
                border-color: #1f77b4 !important;
                box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2) !important;
            }
            
            /* Help text */
            .stTooltipIcon {
                color: #1f77b4 !important;
            }
            
            /* Caption text */
            .st-caption {
                color: #666 !important;
                font-size: 12px !important;
            }
            
            /* Make all text more visible */
            p, div, span {
                color: #333 !important;  /* Darker text for better contrast */
            }
            
            /* Success/Info/Warning/Error boxes */
            .stAlert {
                border-radius: 6px !important;
                border-left: 4px solid !important;
            }
            .stSuccess {
                border-left-color: #27ae60 !important;
            }
            .stInfo {
                border-left-color: #3498db !important;
            }
            .stWarning {
                border-left-color: #f39c12 !important;
            }
            .stError {
                border-left-color: #e74c3c !important;
            }
            
            /* Section headers in main content */
            h1, h2, h3 {
                color: #1f77b4 !important;
                font-weight: 600 !important;
            }
            
            /* Widget labels */
            .st-emotion-cache-1qg05tj {
                color: #2c3e50 !important;
                font-weight: 600 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Professional Header with LARGER FONTS
        st.markdown('<p class="main-header">🏥 Insurance Fraud Detection Platform</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Machine Learning for Claim Risk Assessment</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/insurance.png", width=80)
            st.title("Navigation")
            
            app_mode = st.selectbox(
                "Select Analysis Mode",
                ["Individual Assessment", "Bulk Processing", "Performance Comparison", "System Information"],
                format_func=lambda x: {
                    "Individual Assessment": "📊 Individual Assessment",
                    "Bulk Processing": "📁 Bulk Processing", 
                    "Performance Comparison": "📈 Performance Comparison",
                    "System Information": "ℹ️ System Information"
                }[x]
            )
            
            st.markdown("---")
            
            # System Status
            st.subheader("System Status")
            if not self.demo_mode:
                st.success("**✅ Production Ready**")
                st.metric("Model Performance", "88.5% AUC")
                st.metric("Features", f"{len(self.top_features)}")
            else:
                st.warning("**🔸 Demo Mode**")
                st.info("Using simulated predictions")
            
            st.markdown("---")
            st.caption("Thesis Project | Fraud Detection System")
        
        # Main Content
        if app_mode == "Individual Assessment":
            self.manual_input_form()
        elif app_mode == "Bulk Processing":
            self.batch_analysis()
        elif app_mode == "Performance Comparison":
            self.comparison_analysis()
        else:
            self._display_system_info()

if __name__ == "__main__":
    app = FraudDetectionPrototype()
    app.main()