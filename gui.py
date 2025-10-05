import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFormLayout, QLabel, QSlider, QDoubleSpinBox, QPushButton, 
                             QComboBox, QScrollArea, QGroupBox, QProgressBar, QSplitter,
                             QFrame, QTabWidget, QTextEdit, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import pandas as pd
import joblib
from preprocessing import create_engineered_features
import numpy as np

# Define the original feature column names exactly as in the dataset
features = [
    'Cement (component 1)(kg in a m^3 mixture)',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
    'Fly Ash (component 3)(kg in a m^3 mixture)',
    'Water  (component 4)(kg in a m^3 mixture)',
    'Superplasticizer (component 5)(kg in a m^3 mixture)',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)',
    'Age (day)'
]

# Short names for display
short_names = [
    'Cement',
    'Blast Furnace Slag', 
    'Fly Ash',
    'Water',
    'Superplasticizer',
    'Coarse Aggregate',
    'Fine Aggregate',
    'Age'
]

# Units for display
units = [
    'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'days'
]

# Ranges: (min, max, default) - extended slightly below min and above max
ranges = [
    (90.0, 600.0, 300.0),  # Cement
    (0.0, 400.0, 100.0),   # Slag
    (0.0, 220.0, 50.0),    # Fly Ash
    (110.0, 270.0, 180.0), # Water
    (0.0, 35.0, 5.0),      # Superplasticizer
    (750.0, 1200.0, 1000.0), # Coarse Aggregate
    (550.0, 1050.0, 800.0), # Fine Aggregate
    (1.0, 400.0, 28.0)     # Age
]

class ModernSlider(QWidget):
    def __init__(self, label, unit, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label and value display
        top_layout = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.value_display = QLabel(f"{default_val} {unit}")
        self.value_display.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.value_display.setStyleSheet("color: #007bff; min-width: 80px;")
        self.value_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        top_layout.addWidget(self.label)
        top_layout.addStretch()
        top_layout.addWidget(self.value_display)
        layout.addLayout(top_layout)
        
        # Slider and spinbox
        control_layout = QHBoxLayout()
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val * 10))
        self.slider.setMaximum(int(max_val * 10))
        self.slider.setValue(int(default_val * 10))
        self.slider.setSingleStep(1)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #dee2e6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 2px solid #0056b3;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #0095ff;
                border: 2px solid #007bff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0056b3, stop:1 #007bff);
                border-radius: 3px;
            }
        """)
        
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(1)
        self.spin.setMinimum(min_val)
        self.spin.setMaximum(max_val)
        self.spin.setSingleStep(0.1)
        self.spin.setValue(default_val)
        self.spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
                color: #212529;
                min-width: 70px;
                selection-background-color: #007bff;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 16px;
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 2px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #e9ecef;
            }
        """)
        
        control_layout.addWidget(self.slider, 4)
        control_layout.addWidget(self.spin, 1)
        layout.addLayout(control_layout)
        
        # Min/Max labels
        range_layout = QHBoxLayout()
        min_label = QLabel(f"{min_val}")
        min_label.setStyleSheet("color: #6c757d; font-size: 9px;")
        max_label = QLabel(f"{max_val}")
        max_label.setStyleSheet("color: #6c757d; font-size: 9px;")
        
        range_layout.addWidget(min_label)
        range_layout.addStretch()
        range_layout.addWidget(max_label)
        layout.addLayout(range_layout)
        
        # Connect signals
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin.valueChanged.connect(self.on_spin_changed)
        
    def on_slider_changed(self, value):
        actual_value = value / 10.0
        self.spin.setValue(actual_value)
        self.value_display.setText(f"{actual_value:.1f} {self.value_display.text().split()[-1]}")
        
    def on_spin_changed(self, value):
        self.slider.setValue(int(value * 10))
        self.value_display.setText(f"{value:.1f} {self.value_display.text().split()[-1]}")
        
    def value(self):
        return self.spin.value()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Concrete Strength Predictor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize models dictionary
        self.models = {}
        self.load_models()
        
        # Setup UI
        self.setup_ui()
        
    def load_models(self):
        """Load all models at startup"""
        model_files = {
            'Random Forest': 'saved/models/random_forest.joblib',
            'Extra Trees': 'saved/models/extra_trees.joblib', 
            'LightGBM': 'saved/models/lightgbm.joblib'
        }
        
        for name, path in model_files.items():
            try:
                self.models[name] = joblib.load(path)
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
    def setup_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Input controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Results and details
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])
        
        main_layout.addWidget(splitter)
        
        # Initial calculation
        QTimer.singleShot(100, self.calculate)
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("Concrete Mixture Parameters")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #212529; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Adjust the parameters below to configure your concrete mixture composition:")
        desc.setFont(QFont("Segoe UI", 10))
        desc.setStyleSheet("color: #6c757d; margin-bottom: 20px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Scroll area for sliders
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        
        self.slider_widgets = []
        for i, short_name in enumerate(short_names):
            min_val, max_val, default_val = ranges[i]
            slider = ModernSlider(short_name, units[i], min_val, max_val, default_val)
            self.slider_widgets.append(slider)
            scroll_layout.addWidget(slider)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setStyleSheet(self.get_button_style("#6c757d"))
        reset_btn.clicked.connect(self.reset_to_defaults)
        
        self.calculate_btn = QPushButton("Calculate Strength")
        self.calculate_btn.setStyleSheet(self.get_button_style("#007bff"))
        self.calculate_btn.clicked.connect(self.calculate)
        
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(self.calculate_btn)
        layout.addLayout(button_layout)
        
        return panel
        
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Results section
        results_group = QGroupBox("Prediction Results")
        results_group.setFont(QFont("Segoe UI", 12, QFont.Bold))
        results_group.setStyleSheet("""
            QGroupBox {
                color: #212529;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }
        """)
        results_layout = QVBoxLayout(results_group)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Prediction Model:")
        model_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Random Forest', 'Extra Trees', 'LightGBM'])
        self.model_combo.setFont(QFont("Segoe UI", 10))
        self.model_combo.currentTextChanged.connect(self.calculate)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        results_layout.addLayout(model_layout)
        
        # Strength prediction
        strength_layout = QVBoxLayout()
        strength_label = QLabel("Predicted Compressive Strength")
        strength_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        
        self.strength_value = QLabel("--")
        self.strength_value.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self.strength_value.setStyleSheet("color: #007bff;")
        self.strength_value.setAlignment(Qt.AlignCenter)
        
        unit_label = QLabel("MPa")
        unit_label.setFont(QFont("Segoe UI", 14, QFont.Medium))
        unit_label.setAlignment(Qt.AlignCenter)
        
        strength_layout.addWidget(strength_label)
        strength_layout.addWidget(self.strength_value)
        strength_layout.addWidget(unit_label)
        results_layout.addLayout(strength_layout)
        
        # Strength indicator
        self.strength_bar = QProgressBar()
        self.strength_bar.setRange(0, 100)
        self.strength_bar.setTextVisible(False)
        self.strength_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff4444, stop:0.5 #ffaa00, stop:1 #00ff00);
                border-radius: 3px;
            }
        """)
        results_layout.addWidget(self.strength_bar)
        
        layout.addWidget(results_group)
        
        # Features details in tabs
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 9))
        
        # Original features tab
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)
        self.original_content = QWidget()
        self.original_layout = QFormLayout(self.original_content)
        self.original_scroll.setWidget(self.original_content)
        original_layout.addWidget(self.original_scroll)
        self.tabs.addTab(self.original_tab, "Original Features")
        
        # Engineered features tab  
        self.engineered_tab = QWidget()
        engineered_layout = QVBoxLayout(self.engineered_tab)
        self.engineered_scroll = QScrollArea()
        self.engineered_scroll.setWidgetResizable(True)
        self.engineered_content = QWidget()
        self.engineered_layout = QFormLayout(self.engineered_content)
        self.engineered_scroll.setWidget(self.engineered_content)
        engineered_layout.addWidget(self.engineered_scroll)
        self.tabs.addTab(self.engineered_tab, "Engineered Features")
        
        layout.addWidget(self.tabs)
        
        return panel
        
    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color)};
            }}
        """
        
    def lighten_color(self, hex_color):
        # Simple color lightening for hover effect
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def darken_color(self, hex_color):
        # Simple color darkening for pressed effect
        r = max(0, int(hex_color[1:3], 16) - 30)
        g = max(0, int(hex_color[3:5], 16) - 30)
        b = max(0, int(hex_color[5:7], 16) - 30)
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def reset_to_defaults(self):
        """Reset all sliders to default values"""
        for i, slider_widget in enumerate(self.slider_widgets):
            min_val, max_val, default_val = ranges[i]
            slider_widget.spin.setValue(default_val)
        self.calculate()
        
    def calculate(self):
        try:
            # Collect input values
            input_values = [slider.value() for slider in self.slider_widgets]
            input_dict = dict(zip(features, input_values))
            input_df = pd.DataFrame([input_dict])

            # Compute engineered features
            engineered_df = create_engineered_features(input_df)

            # Update feature displays
            self.update_feature_displays(input_dict, engineered_df)

            # Handle categorical encoding if needed
            categorical_columns = engineered_df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                encoder_path = 'saved/preprocessing/encoder.joblib'
                encoder = joblib.load(encoder_path)
                engineered_df[categorical_columns] = encoder.transform(engineered_df[categorical_columns])

            # Load selected model and predict
            selected_model = self.model_combo.currentText()
            if selected_model in self.models:
                model = self.models[selected_model]
                prediction = model.predict(engineered_df)[0]
                
                # Update display
                self.strength_value.setText(f"{prediction:.1f}")
                
                # Update progress bar (assuming typical concrete strength range 0-100 MPa)
                strength_percent = min(100, max(0, int(prediction)))
                self.strength_bar.setValue(strength_percent)
                
            else:
                self.strength_value.setText("Error")
                
        except Exception as e:
            self.strength_value.setText("Error")
            print(f"Calculation error: {e}")
            
    def update_feature_displays(self, original_features, engineered_df):
        # Clear previous displays
        for i in reversed(range(self.original_layout.count())):
            item = self.original_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
                
        for i in reversed(range(self.engineered_layout.count())):
            item = self.engineered_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
                
        # Display original features
        for feature, value in original_features.items():
            # Create a nicer display name
            display_name = feature.replace('(component 1)', '')\
                                 .replace('(component 2)', '')\
                                 .replace('(component 3)', '')\
                                 .replace('(component 4)', '')\
                                 .replace('(component 5)', '')\
                                 .replace('(component 6)', '')\
                                 .replace('(component 7)', '')\
                                 .replace('(kg in a m^3 mixture)', '')\
                                 .replace('  ', ' ').strip()
                                 
            label = QLabel(display_name)
            label.setStyleSheet("font-size: 11px; color: #495057; padding: 2px;")
            
            value_label = QLabel(f"{value:.1f}")
            value_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #007bff; padding: 2px;")
            
            self.original_layout.addRow(label, value_label)
            
        # Display engineered features
        for col in engineered_df.columns:
            if col not in original_features:  # Only show engineered features
                value = engineered_df[col].iloc[0]
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                    
                label = QLabel(col)
                label.setStyleSheet("font-size: 11px; color: #495057; padding: 2px;")
                
                value_label = QLabel(value_str)
                value_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #28a745; padding: 2px;")
                
                self.engineered_layout.addRow(label, value_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Modern light theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
    palette.setColor(QPalette.Base, QColor(248, 249, 250))
    palette.setColor(QPalette.AlternateBase, QColor(233, 236, 239))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(33, 37, 41))
    palette.setColor(QPalette.Text, QColor(33, 37, 41))
    palette.setColor(QPalette.Button, QColor(233, 236, 239))
    palette.setColor(QPalette.ButtonText, QColor(33, 37, 41))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(0, 123, 255))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ffffff;
        }
        QWidget {
            background-color: #ffffff;
            color: #212529;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QScrollArea {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        QComboBox {
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 5px 10px;
            color: #212529;
            selection-background-color: #007bff;
            min-width: 120px;
        }
        QComboBox::drop-down {
            border-left: 1px solid #ced4da;
            width: 20px;
        }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #212529;
            selection-background-color: #007bff;
            border: 1px solid #ced4da;
        }
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        QTabBar::tab {
            background-color: #e9ecef;
            color: #495057;
            padding: 8px 12px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #007bff;
            color: white;
        }
        QTabBar::tab:hover:!selected {
            background-color: #dee2e6;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())