from flask import Flask, request, render_template_string, jsonify
from model import SystemManager
import random
from pygments.lexers import guess_lexer

app = Flask(__name__)

# Instantiate and start the system
system_manager = SystemManager()
system_manager.start_all()

# --------------------- Home Page Route ---------------------
@app.route("/")
def index():
    # A creative, light-themed UI with separate sections for Error Report, Optimization Suggestions, and Optimized Code.
    html = """
    <!doctype html>
    <html>
    <head>
      <title>Opticode Ai</title>
      <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,400,600" rel="stylesheet">
      <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
      <style>
        body {
          font-family: 'Open Sans', sans-serif;
          background: #f2f2f2;
          color: #333;
          margin: 0;
          padding: 20px;
        }
        #container {
          max-width: 1200px;
          margin: auto;
          background: #ffffff;
          padding: 30px;
          border-radius: 10px;
          box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        h1 {
          text-align: center;
          color: #2c3e50;
          margin-bottom: 20px;
        }
        label {
          font-size: 16px;
          font-weight: 600;
          margin-bottom: 5px;
          display: block;
        }
        #lang_select {
          padding: 10px;
          font-size: 16px;
          margin-bottom: 20px;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
        #code {
          width: 100%;
          height: 250px;
          font-family: 'Courier New', monospace;
          font-size: 14px;
          border: 1px solid #ccc;
          border-radius: 4px;
          padding: 10px;
          box-sizing: border-box;
          resize: vertical;
          margin-bottom: 20px;
        }
        .btn {
          padding: 10px 20px;
          background: #3498db;
          border: none;
          border-radius: 4px;
          color: #fff;
          cursor: pointer;
          font-size: 16px;
          margin-right: 10px;
          margin-bottom: 20px;
        }
        .btn:hover {
          background: #2980b9;
        }
        .section {
          background: #ecf0f1;
          padding: 20px;
          border-radius: 4px;
          margin-top: 20px;
          border: 1px solid #bdc3c7;
        }
        .section h2 {
          margin-top: 0;
          font-size: 20px;
          color: #16a085;
          border-bottom: 1px solid #bdc3c7;
          padding-bottom: 10px;
        }
        pre {
          white-space: pre-wrap;
          word-wrap: break-word;
          font-family: 'Courier New', monospace;
          font-size: 14px;
          margin: 0;
        }
      </style>
    </head>
    <body>
      <div id="container">
        <h1>Opticode Ai - Code Analyzer & Optimizer</h1>
        <label for="lang_select">Select Language:</label>
        <select id="lang_select">
          <option value="Python">Python</option>
          <option value="Java">Java</option>
          <option value="C">C</option>
          <option value="C++">C++</option>
          <option value="Verilog">Verilog</option>
        </select>
        <br/>
        <label for="code">Enter your code:</label>
        <textarea id="code" placeholder="Type your code here..."></textarea>
        <button class="btn" id="analyzeBtn">Analyze Code</button>
        <div class="section" id="errorSection">
          <h2>Error Report</h2>
          <div id="error_report">Error report will appear here.</div>
        </div>
        <div class="section" id="suggestionsSection">
          <h2>Optimization Suggestions</h2>
          <div id="suggestions">Suggestions will appear here.</div>
        </div>
        <div class="section" id="optimizedCodeSection">
          <h2>Optimized Code</h2>
          <div id="optimized_code">Optimized code will appear here.</div>
        </div>
      </div>
      <script>
        // Standard analysis: error report, suggestions, and optimized code
        function analyzeCode(){
          var code = $("#code").val();
          var language = $("#lang_select").val();
          $.post("/analyze", {code: code, language: language}, function(data){
            $("#error_report").html("<pre>" + data.error_report + "</pre>");
            $("#suggestions").html("<pre>" + data.suggestions + "</pre>");
            $("#optimized_code").html("<pre>" + data.optimized_code + "</pre>");
          });
        }
        // Button click event
        $("#analyzeBtn").on("click", function(){
          analyzeCode();
        });
        // Auto-analyze on input change (debounced)
        var timer;
        $("#code").on("input", function(){
          clearTimeout(timer);
          timer = setTimeout(function(){
            analyzeCode();
          }, 500);
        });
        // Update language setting on dropdown change
        $("#lang_select").on("change", function(){
          var language = $(this).val();
          $.post("/set_language", {language: language}, function(data){
            console.log("Language set to " + data.language);
          });
        });
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

# --------------------- Set Language Route ---------------------
@app.route("/set_language", methods=["POST"])
def set_language():
    language = request.form.get("language", "Python")
    validated_language = system_manager.language_manager.validate_language(language)
    system_manager.language_manager.set_default_language(validated_language)
    return jsonify({"language": validated_language})

# --------------------- Analyze Route ---------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    code = request.form.get("code", "")
    language = request.form.get("language", "")
    if not code:
        return jsonify({
            "language": language,
            "error_report": "No code provided.",
            "suggestions": "",
            "optimized_code": ""
        })
    validated_language = system_manager.language_manager.validate_language(language)
    try:
        lexer = guess_lexer(code)
        detected_language = lexer.name
    except Exception:
        detected_language = validated_language
    basic_errors = system_manager.llm.detect_errors(validated_language, code)
    adv_errors = system_manager.error_checker.check_errors(validated_language, code)
    enhanced_errors = system_manager.error_checker.enhanced_error_report(validated_language, code)
    error_report = ("Syntax Errors (Basic):\n" + basic_errors + "\n\n" +
                    "Advanced Error Check:\n" + adv_errors + "\n\n" +
                    "Enhanced Error Report:\n" + enhanced_errors)
    simulated_runtime = random.randint(500, 1500)
    detailed = system_manager.llm.get_detailed_suggestion(validated_language, code, simulated_runtime)
    return jsonify({
        "language": validated_language,
        "error_report": error_report,
        "suggestions": detailed["suggestions"],
        "optimized_code": detailed["optimized_code"]
    })

# --------------------- Additional Utility Routes ---------------------
@app.route("/status")
def status():
    status = system_manager.get_system_status()
    return jsonify(status)

@app.route("/report")
def report():
    full_report = system_manager.generate_full_report()
    return "<pre>" + full_report + "</pre>"

@app.route("/dashboard")
def dashboard():
    summary = system_manager.dashboard.get_summary()
    return jsonify(summary)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
