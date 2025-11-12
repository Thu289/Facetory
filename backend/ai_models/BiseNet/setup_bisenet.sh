#!/bin/bash
# Setup script for BiSeNet Face Parsing
# Clones the official repository and sets up the model
# Follows the path structure used in inference_bisenet.py

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BISENET_REPO_DIR="$SCRIPT_DIR/face-parsing.PyTorch"

echo "🔧 Setting up BiSeNet Face Parsing..."

# Clone repository if not exists
if [ ! -d "$BISENET_REPO_DIR" ]; then
    echo "📥 Cloning BiSeNet repository..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/zllrunning/face-parsing.PyTorch.git
    echo "✅ Repository cloned"
else
    echo "✅ Repository already exists: $BISENET_REPO_DIR"
fi

# Check for weights in both locations (as per inference_bisenet.py)
# Path 1: Official repo location
REPO_WEIGHTS_DIR="$BISENET_REPO_DIR/res/cp"
REPO_WEIGHTS_FILE="$REPO_WEIGHTS_DIR/79999_iter.pth"

# Path 2: Fallback location (pretrained_models/bisenet)
FALLBACK_WEIGHTS_DIR="$SCRIPT_DIR/pretrained_models/bisenet"
FALLBACK_WEIGHTS_FILE="$FALLBACK_WEIGHTS_DIR/79999_iter.pth"

echo ""
echo "📁 Checking for pre-trained weights..."

# Check official repo location first
if [ -f "$REPO_WEIGHTS_FILE" ]; then
    echo "✅ Pre-trained weights found in official repo location:"
    echo "   $REPO_WEIGHTS_FILE"
    WEIGHTS_FOUND=true
elif [ -f "$FALLBACK_WEIGHTS_FILE" ]; then
    echo "✅ Pre-trained weights found in fallback location:"
    echo "   $FALLBACK_WEIGHTS_FILE"
    WEIGHTS_FOUND=true
else
    echo "⚠️  Pre-trained weights not found in either location"
    echo ""
    echo "📥 Please download weights manually from:"
    echo "   https://drive.google.com/file/d/1a1_0xT5YQYfNU3IKH77HX4sNm9X0jY0E/view"
    echo ""
    echo "💡 You can save the file to either location:"
    echo "   Option 1 (Official repo): $REPO_WEIGHTS_FILE"
    echo "   Option 2 (Fallback):      $FALLBACK_WEIGHTS_FILE"
    echo ""
    
    # Create directories if they don't exist
    mkdir -p "$REPO_WEIGHTS_DIR"
    mkdir -p "$FALLBACK_WEIGHTS_DIR"
    
    echo "📂 Created directories for weights:"
    echo "   - $REPO_WEIGHTS_DIR"
    echo "   - $FALLBACK_WEIGHTS_DIR"
    WEIGHTS_FOUND=false
fi

echo ""
echo "✅ Setup complete!"
echo ""

if [ "$WEIGHTS_FOUND" = false ]; then
    echo "📋 Next steps:"
    echo "   1. Download the weights file (79999_iter.pth) from the link above"
    echo "   2. Save it to one of the directories shown above"
    echo "   3. The model can then be used via inference_bisenet.py"
else
    echo "✅ BiSeNet is ready to use!"
    echo "   Run inference_bisenet.py or use via API endpoint /api/face/makeup/style_extract"
fi

echo ""
echo "📚 Model files location:"
echo "   Repository:  $BISENET_REPO_DIR"
if [ -f "$REPO_WEIGHTS_FILE" ]; then
    echo "   Weights:     $REPO_WEIGHTS_FILE"
elif [ -f "$FALLBACK_WEIGHTS_FILE" ]; then
    echo "   Weights:     $FALLBACK_WEIGHTS_FILE"
fi

