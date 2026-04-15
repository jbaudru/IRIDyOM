// MIDI to Note Name Converter for Max
// Usage: [number] → [js midi_to_note.js] → [outlet]

// This function is called automatically when a number is sent to the js object
function msg_int(num) {
    var notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    var octave = Math.floor(num / 12) - 1;
    var note = notes[num % 12];
    outlet(0, note + octave);
}

// Also handle float input
function msg_float(num) {
    msg_int(Math.floor(num));
}