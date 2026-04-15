/**
 * jsui_ic_plot.js - JSUI object for plotting user_note_ic in real-time
 * 
 * INLET 0: float - IC value in bits (0-10 range)
 *          "clear" - clears all plot data
 *          "reset_max" - resets the maximum IC value for rescaling
 * OUTLETS: None (graphics only)
 */

inlets = 1;
outlets = 0;

var ic_data = [];
var max_samples = 50; 
var max_ic_value = 1.0;  // Track maximum IC value for normalization

/**
 * Handle incoming float values - DRAW HERE directly
 */
function msg_float(value) {
    post("msg_float: " + value + "\n");
    value = parseFloat(value);
    ic_data.push(value);
    if (ic_data.length > max_samples) {
        ic_data.shift();
    }
    
    // Update max IC value for dynamic scaling
    if (value > max_ic_value) {
        max_ic_value = value;
    }
    
    post("IC data: " + value.toFixed(3) + " bits (" + ic_data.length + " samples, max: " + max_ic_value.toFixed(2) + ")\n");
    
    draw_plot();
}

/**
 * Draw the complete plot
 */
function draw_plot() {
    // DRAW PLOT
    sketch.glclearcolor(0.05, 0.05, 0.1, 0);  // Transparent background
    sketch.glclear();  // Clear with transparent color
    
    // === AXES ===
    sketch.glcolor(0.5, 0.5, 0.5);  // Gray
    sketch.gllinewidth(0.8);
    sketch.moveto(-1.15, -0.8);         // Bottom-left with margin
    sketch.lineto(-1.15, 0.9);          // Left axis (Y)
    sketch.moveto(-1.15, -0.8);
    sketch.lineto(1.0, -0.8);          // Bottom axis (X)
    
    // === GRID LINES ===
    sketch.glcolor(0.3, 0.3, 0.4, 0.1);  // Lighter grid (reduced opacity)
    sketch.gllinewidth(0.4);
    
    // Horizontal grid lines (for IC values 0-10)
    for (var i = 1; i < 10; i++) {
        var y = -0.8 + (i / 10) * 1.7;  // 0-10 bits scaled
        sketch.moveto(-1.15, y);
        sketch.lineto(1.0, y);
    }
    
    // Vertical grid lines (for time steps)
    for (var i = 1; i < 10; i++) {
        var x = -1.15 + (i / 10) * 2.15;  // Time scaled (wider)
        sketch.moveto(x, -0.8);
        sketch.lineto(x, 0.9);
    }
    
    // === IC DATA LINE (GREEN) ===
    if (ic_data.length > 1) {
        sketch.glcolor(0.0, 1.0, 0.0, 0.6);  // Bright green, semi-transparent
        sketch.gllinewidth(1.8);
        
        for (var i = 0; i < ic_data.length; i++) {
            var x = -1.15 + (i / max_samples) * 2.15;             // X: time scaled (wider)
            var normalized_y = Math.min(1.0, Math.max(0.0, ic_data[i] / (max_ic_value || 1.0)));  // Normalize to 0-1 range
            var y = -0.8 + normalized_y * 1.7;                  // Y: scaled
            
            if (i === 0) {
                sketch.moveto(x, y);
            } else {
                sketch.lineto(x, y);
            }
        }
    }
    
    // === DATA POINTS (GREEN CIRCLES) ===
    if (ic_data.length > 0) {
        sketch.glcolor(0.2, 1.0, 0.6, 0.4);  // Bright green, semi-transparent
        for (var i = 0; i < ic_data.length - 1; i++) {
            var x = -1.15 + (i / max_samples) * 2.15;
            var normalized_y = Math.min(1.0, Math.max(0.0, ic_data[i] / (max_ic_value || 1.0)));  // Normalize to 0-1 range
            var y = -0.8 + normalized_y * 1.7;
            sketch.moveto(x, y);
            sketch.circle(0.018);
        }
        
        // === LATEST PLAYED DOT (BRIGHT GREEN) ===
        sketch.glcolor(0.0, 1.0, 0.0, 1.0);  // Bright opaque green
        var last_i = ic_data.length - 1;
        var last_x = -1.15 + (last_i / max_samples) * 2.15;
        var last_normalized_y = Math.min(1.0, Math.max(0.0, ic_data[last_i] / (max_ic_value || 1.0)));  // Normalize to 0-1 range
        var last_y = -0.8 + last_normalized_y * 1.7;
        sketch.moveto(last_x, last_y);
        sketch.circle(0.045);  // Larger dot for latest
    }
    
    // === AXIS LABELS ===
    sketch.glcolor(0.8, 0.8, 0.8);  // Light gray text
    sketch.fontsize(8);
    sketch.moveto(-1.15, -1.0);
    sketch.text("0");
    sketch.moveto(1.0, -1.0);
    sketch.text("t");
    
    // === LATEST IC VALUE ===
    if (ic_data.length > 0) {
        var last = ic_data[ic_data.length - 1];
        sketch.glcolor(0.9, 0.9, 1.0);  // Bright text
        sketch.fontsize(7);
        sketch.moveto(-1.05, 0.85);
        sketch.text("IC: " + last.toFixed(2));
    }
    
    refresh();
}

/**
 * Handle incoming messages (e.g., "clear", "reset_max")
 */
function msg_anything(args) {
    post("msg_anything received: " + args.join(", ") + "\n");
    var msg = args[0];
    
    if (msg === "clear") {
        post("CLEARING PLOT\n");
        ic_data = [];
        max_ic_value = 1.0;
        post("IC plot cleared\n");
        draw_plot();  // Redraw empty plot
    } 
    else if (msg === "reset_max") {
        max_ic_value = 1.0;
        post("IC max value reset to 1.0\n");
        draw_plot();  // Redraw with new scaling
    }
    else {
        post("Unknown message: " + msg + "\n");
    }
}

/**
 * Direct handler for "clear" symbol
 */
function clear() {
    post("CLEAR HANDLER CALLED\n");
    ic_data = [];
    max_ic_value = 1.0;
    post("IC plot cleared via clear()\n");
    draw_plot();  // Redraw empty plot
}

/**
 * Direct handler for "reset_max" symbol
 */
function reset_max() {
    post("RESET_MAX HANDLER CALLED\n");
    max_ic_value = 1.0;
    post("IC max value reset to 1.0 via reset_max()\n");
    draw_plot();  // Redraw with new scaling
}

/**
 * Direct handler for "bang"
 */
function bang() {
    post("BANG HANDLER CALLED\n");
    ic_data = [];
    max_ic_value = 1.0;
    post("IC plot cleared via bang\n");
    draw_plot();  // Redraw empty plot
}
