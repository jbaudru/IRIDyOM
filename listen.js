// Node.js TCP client for connecting to the Python MIDI TCP server.
// This script is intended to run inside Max's `node.script` (Node) object or standalone Node.
// If loaded into the legacy `js` object this file will bail out with a helpful message.

var HOST = '127.0.0.1';
var PORT = 5005;

// detect Node environment
var inNode = (typeof process !== 'undefined' && process.versions && process.versions.node);

// helper log/post functions
var log = function() { if (typeof console !== 'undefined' && console.log) console.log.apply(console, arguments); };
var error = function() { if (typeof console !== 'undefined' && console.error) console.error.apply(console, arguments); };
if (!inNode) {
	// running in legacy Max `js` object — inform the user and stop to avoid syntax/runtime errors
	if (typeof post === 'function') post('listen.js: please use a node.script object (Node) not the legacy js object.\n');
	else log('listen.js: please use a node.script object (Node) not the legacy js object.');
	// no further action
} else {
	var net = require('net');
	var path = require('path');
	var fileURLToPath = null;
	try {
		fileURLToPath = require('url').fileURLToPath;
	} catch (e) {
		// Older Max Node runtimes may not expose fileURLToPath.
	}
	var maxApi = null;
	try {
		maxApi = require('max-api');
		if (maxApi && typeof maxApi.post === 'function') maxApi.post('max-api available: will forward messages to Max');
		else log('max-api available: will forward messages to Max');
	} catch (e) {
		log('max-api not available; will print messages to stdout');
		log('Error loading max-api: ' + (e.message || e));
	}

	// Use maxApi.post when available so messages appear in Max Console reliably
	// In Ableton Live, maxApi may not be available - that's OK, we'll use console.log
	var poster = (maxApi && typeof maxApi.post === 'function') ? maxApi.post : log;

	// Reconnection logic: create a fresh socket on each attempt
	var client = null;
	var buffer = '';
	var reconnectDelay = 500; // ms
	var maxReconnect = 10000;
	var reconnectTimer = null;
	var isAbletonLive = !maxApi || typeof maxApi.post !== 'function'; // Detect if running in Ableton Live
	
	// Track currently playing note to prevent chords
	var currentlyPlayingNote = null;
	var currentNoteEndTime = 0;  // When the current note is supposed to end (timestamp in ms)
	var minGapBetweenNotes = 10; // milliseconds - minimum gap after note ends before next note starts
	var latestSequencerPhase = null;
	
	// Queue for pending notes to prevent duplicates and ensure strict sequencing
	var pendingNoteQueue = [];  // Array of {pitch, vel, durationMs}
	var scheduledNoteTimer = null;  // Tracks the setTimeout for the next pending note

	if (isAbletonLive) {
		log('=== Running in Ableton Live (no max-api) ===');
	} else {
		log('=== Running in Max (max-api available) ===');
	}

	// Function to process the next pending note
	function processPendingNotes() {
		if (pendingNoteQueue.length === 0) {
			scheduledNoteTimer = null;
			return;
		}
		
		var now = Date.now();
		var timeUntilNoteCanStart = currentNoteEndTime - now;
		
		if (timeUntilNoteCanStart > 0) {
			// Still waiting for current note to end
			var delayNeeded = timeUntilNoteCanStart + minGapBetweenNotes;
			scheduledNoteTimer = setTimeout(processPendingNotes, delayNeeded);
			return;
		}
		
		// Ready to send the next note
		var nextNote = pendingNoteQueue.shift();  // Remove from queue
		var pitch = nextNote.pitch;
		var vel = nextNote.vel;
		var durationMs = nextNote.durationMs;
		
		currentlyPlayingNote = pitch;
		currentNoteEndTime = now + durationMs;
		maxApi.outlet(pitch, vel, durationMs);
		poster('Sent queued note: pitch=' + pitch + ', vel=' + vel + ', duration=' + durationMs + 'ms');
		
		// Schedule next note if queue not empty
		if (pendingNoteQueue.length > 0) {
			var nextDelayNeeded = durationMs + minGapBetweenNotes;
			scheduledNoteTimer = setTimeout(processPendingNotes, nextDelayNeeded);
		} else {
			scheduledNoteTimer = null;
		}
	}
	function sendParameter(paramObj) {
		if (client && !client.destroyed) {
			try {
				var msg = JSON.stringify(paramObj) + '\n';
				client.write(msg);
				poster('Sent parameter: ' + JSON.stringify(paramObj));
			} catch (e) {
				error('Failed to send parameter:', e);
			}
		} else {
			error('Cannot send parameter: not connected');
		}
	}

	function parseFiniteNumber(value) {
		var parsed = parseFloat(value);
		return isFinite(parsed) ? parsed : null;
	}

	function addLiveNoteArgs(params, args, startIndex) {
		var velocity = parseFiniteNumber(args[startIndex]);
		if (velocity !== null) params.velocity = velocity;

		var third = parseFiniteNumber(args[startIndex + 1]);
		var fourth = parseFiniteNumber(args[startIndex + 2]);
		if (third !== null) {
			if (fourth !== null) {
				params.duration_ms = third;
				params.channel = fourth;
			} else if (Math.floor(third) === third && third >= 0 && third <= 16) {
				params.channel = third;
			} else {
				params.duration_ms = third;
			}
		}
	}

	function sendLiveNote(selector, args) {
		if (!args || args.length === 0) return;
		var note = parseInt(args[0]);
		if (!isFinite(note)) return;

		var params = {};
		params[selector] = note;
		params.client_time_ms = Date.now();
		addLiveNoteArgs(params, args, 1);
		sendParameter(params);
	}

	function storeSequencerPhase(args) {
		var values = Array.prototype.slice.call(args || []);
		latestSequencerPhase = {
			bar: parseFiniteNumber(values[0]),
			beat: parseFiniteNumber(values[1]),
			unit: parseFiniteNumber(values[2]),
			ticks: parseFiniteNumber(values[3]),
			tempo: parseFiniteNumber(values[4]),
			client_time_ms: Date.now()
		};
	}

	function flattenArgs(args) {
		var flattened = [];
		function add(value) {
			if (Array.isArray(value)) {
				for (var i = 0; i < value.length; i++) add(value[i]);
			} else if (value !== undefined && value !== null) {
				flattened.push(String(value));
			}
		}
		for (var i = 0; i < args.length; i++) add(args[i]);
		return flattened;
	}

	function stripWrappingQuotes(text) {
		text = String(text || '').trim();
		while (text.length >= 2) {
			var first = text.charAt(0);
			var last = text.charAt(text.length - 1);
			if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
				text = text.slice(1, -1).trim();
			} else {
				break;
			}
		}
		return text;
	}

	function normalizePathText(text) {
		var raw = stripWrappingQuotes(text);
		if (!raw) return raw;

		if (/^file:\/\//i.test(raw)) {
			try {
				if (fileURLToPath) {
					raw = fileURLToPath(raw);
				} else {
					raw = decodeURIComponent(raw.replace(/^file:\/\/localhost/i, '').replace(/^file:\/\//i, ''));
				}
			} catch (e) {
				raw = raw.replace(/^file:\/\/localhost/i, '').replace(/^file:\/\//i, '');
				try {
					raw = decodeURIComponent(raw);
				} catch (decodeError) {}
			}
		} else if (/%[0-9A-Fa-f]{2}/.test(raw)) {
			try {
				raw = decodeURIComponent(raw);
			} catch (e) {}
		}

		raw = raw.replace(/\\ /g, ' ');
		return path.normalize(raw);
	}

	function normalizePathArgs(args) {
		return normalizePathText(flattenArgs(args).join(' '));
	}

	function isStrategyAtom(value) {
		return /^[0-2]$/.test(String(value).trim());
	}

	function parseStrategyText(value) {
		var values = String(value || '').replace(/,/g, ' ').trim().split(/\s+/).filter(Boolean);
		if (values.length !== 8 && values.length !== 9) return null;
		for (var i = 0; i < values.length; i++) {
			if (!isStrategyAtom(values[i])) return null;
		}
		return values;
	}

	function splitPathAndStrategies(args) {
		var parts = flattenArgs(args);
		var strategies = [];
		if (parts.length > 1) {
			var parsedTextStrategies = parseStrategyText(parts[parts.length - 1]);
			if (parsedTextStrategies) {
				parts.pop();
				return {
					path: normalizePathArgs(parts),
					strategies: parsedTextStrategies
				};
			}
		}
		while (parts.length > 0 && isStrategyAtom(parts[parts.length - 1])) {
			strategies.unshift(parts.pop());
		}
		if (strategies.length !== 8 && strategies.length !== 9) {
			parts = parts.concat(strategies);
			strategies = [];
		}
		return {
			path: normalizePathArgs(parts),
			strategies: strategies
		};
	}

	function splitPathAndTrailingArgs(args, trailingCount) {
		var parts = flattenArgs(args);
		var trailing = [];
		while (trailing.length < trailingCount && parts.length > 0) {
			trailing.unshift(parts.pop());
		}
		return {
			path: normalizePathArgs(parts),
			trailing: trailing
		};
	}

	function createAndConnect() {
		if (client && !client.destroyed) return;
		log('=== Attempting to create socket and connect to ' + HOST + ':' + PORT + ' ===');
		client = new net.Socket();
		
		// Set connection timeout for M4L environments
		client.setTimeout(5000); // 5 second timeout

		client.connect(PORT, HOST, function() {
			log('=== CONNECTION ESTABLISHED ===');
			poster('Connected to MIDI TCP server');
			client.setTimeout(0); // Disable idle timeout after a successful connection.
			reconnectDelay = 500; // reset backoff
		});

		client.on('data', function(data) {
			buffer += data.toString('utf8');
			var lines = buffer.split('\n');
			buffer = lines.pop(); // last partial
			for (var i = 0; i < lines.length; i++) {
				var line = lines[i];
				if (!line.trim()) continue;
				try {
					var msg = JSON.parse(line);
					if (maxApi) {
						// For Max: output pitch, velocity, and duration in milliseconds
						// Connect to [unpack i i i] then to [makenote] in Max
						if (msg.type === 'midi' && msg.cmd === 'note_on') {
							var pitch = Math.floor(msg.note);
							var vel = msg.vel ? Math.max(0, Math.min(127, Math.floor(msg.vel))) : 100;
							var durationMs = msg.duration ? Math.max(1, Math.round(msg.duration * 1000)) : 500;
							
							var now = Date.now();
							var timeUntilNoteCanStart = currentNoteEndTime - now;
							
							// If current note still playing, queue this note
							if (timeUntilNoteCanStart > 0) {
								pendingNoteQueue.push({pitch: pitch, vel: vel, durationMs: durationMs});
								poster('Queued note (current ends in ' + Math.round(timeUntilNoteCanStart) + 'ms): pitch=' + pitch);
								
								// Schedule processing if not already scheduled
								if (!scheduledNoteTimer) {
									var delayNeeded = timeUntilNoteCanStart + minGapBetweenNotes;
									scheduledNoteTimer = setTimeout(processPendingNotes, delayNeeded);
								}
							} else {
								// Safe to send immediately
								currentlyPlayingNote = pitch;
								currentNoteEndTime = now + durationMs;
								maxApi.outlet(pitch, vel, durationMs);
								poster('Note on: pitch=' + pitch + ', vel=' + vel + ', duration=' + durationMs + 'ms');
							}
						} else if (msg.type === 'midi' && msg.cmd === 'note_off') {
							// Note_off messages: just log them (duration is managed by durationMs)
							var pitch = Math.floor(msg.note);
							poster('Note off received: pitch=' + pitch + ' (duration-based timing in use)');
						} else if (msg.type === 'models_list') {
					// Send models to umenu
					// umenu accepts: clear, then "append item" for each item
						var models = msg.models || [];
						
						// Clear the menu first
						maxApi.outlet('clear');
						
						// Add each model - replace "/" with " - " for cleaner display
						for (var j = 0; j < models.length; j++) {
							var displayName = models[j].name.replace(/\//g, ' - ');
							maxApi.outlet('append', displayName);
						}
						poster('Loaded ' + models.length + ' models into menu');
						} else if (msg.type === 'current_model') {
						// Output current model index for live.menu selection
							if (msg.model && typeof msg.model.index !== 'undefined') {
								maxApi.outlet('set', msg.model.index);
								poster('Current model: ' + msg.model.name);
							}
						} else if (msg.type === 'predictions') {
						// Output user note IC with label for Max route object
						if (msg.user_note_ic !== undefined && msg.user_note_ic !== null) {
							maxApi.outlet('user_note_ic', msg.user_note_ic);
							poster('Sent IC to route: ' + msg.user_note_ic.toFixed(3) + ' bits');
						}
						
						// Output top 3 predictions separately
						// Format: predictions [note1 prob1] [note2 prob2] [note3 prob3]
						var preds = msg.predictions || [];
						var predArray = ['predictions'];
						for (var k = 0; k < preds.length; k++) {
							predArray.push(preds[k].note);
							predArray.push(preds[k].prob);
						}
						maxApi.outlet(...predArray);
						
						var userIC = msg.user_note_ic !== undefined ? msg.user_note_ic : null;
						var userProb = msg.user_note_prob !== undefined ? msg.user_note_prob : null;
						var icStr = userIC !== null ? ' (IC: ' + userIC.toFixed(3) + ' bits, prob: ' + (userProb * 100).toFixed(1) + '%)' : '';
						poster('Predictions for note ' + msg.user_note + icStr + ': ' + JSON.stringify(preds));
						} else if (msg.type === 'train_ack') {
							// Training note acknowledgment
							maxApi.outlet('train_ack', msg.note, msg.sequence_length);
							poster('Train ack: note ' + msg.note + ', sequence length: ' + msg.sequence_length);
						} else if (msg.type === 'train_status') {
							// Training status updates
							maxApi.outlet('train_status', msg.status, JSON.stringify(msg));
							poster('Train status: ' + msg.status + ' - ' + (msg.message || ''));
						} else if (msg.type === 'training_file_status') {
							// File added/removed status - output just the file count
							var totalFiles = msg.total_files || 0;
							maxApi.outlet('file_count', totalFiles);
							poster('Training files: ' + totalFiles);
						} else if (msg.type === 'training_result') {
							// Training completion status
							// Format: training_result <status> <message> <dataset_name>
							var tStatus = msg.status || 'unknown';
							var tMessage = msg.message || '';
							var datasetName = msg.dataset_name || '';
							maxApi.outlet('training_result', tStatus, tMessage, datasetName);
							poster('Training result: ' + tStatus + ' - ' + tMessage);
						} else if (msg.type === 'training_option_set') {
							// Confirmation that option was set
							maxApi.outlet('training_option_set', msg.option, String(msg.value));
							poster('Training option set: ' + msg.option + ' = ' + msg.value);
						} else {
							maxApi.outlet(JSON.stringify(msg));
						}
					} else {
						// stdout fallback
						if (msg.type === 'midi' && msg.cmd === 'note_on') {
							var p = Math.floor(msg.note);
							var v = msg.vel ? Math.max(0, Math.min(127, Math.floor(msg.vel))) : 100;
							var d = msg.duration ? Math.max(1, Math.round(msg.duration * 1000)) : 500;
							poster(p + ' ' + v + ' ' + d);
						}
					}
				} catch (err) {
					error('Failed to parse message:', err, line);
				}
			}
		});

		client.on('close', function() {
			poster('Connection closed');
			currentlyPlayingNote = null;
			currentNoteEndTime = 0;
			pendingNoteQueue = [];  // Clear queue
			if (scheduledNoteTimer) {
				clearTimeout(scheduledNoteTimer);
				scheduledNoteTimer = null;
			}
			scheduleReconnect();
		});

		client.on('error', function(err) {
			error('Socket error', err && err.message ? err.message : err);
			log('=== CONNECTION ERROR: ' + (err && err.message ? err.message : err) + ' ===');
			// ensure socket is destroyed so reconnect creates fresh socket
			try { client.destroy(); } catch (e) {}
		});

		client.on('timeout', function() {
			log('=== CONNECTION TIMEOUT ===');
			error('Socket timeout - server may not be running');
			try { client.destroy(); } catch (e) {}
		});
	}

	function scheduleReconnect() {
		if (reconnectTimer) return;
		reconnectTimer = setTimeout(function() {
			reconnectTimer = null;
			reconnectDelay = Math.min(maxReconnect, Math.round(reconnectDelay * 1.5));
			poster('Attempting reconnect...');
			createAndConnect();
		}, reconnectDelay);
	}

	// start initial connection
	log('=== listen.js: Starting initial connection ===');
	log('Attempting to connect to ' + HOST + ':' + PORT);
	createAndConnect();

	// If running as a node.script inside Max, expose handlers to allow Max to request reconnect/stop
	if (maxApi) {
		poster('=== Registering handlers with max-api ===');
		
		try {
			maxApi.addHandler('reconnect', function() {
				try {
					if (client && client.destroyed) createAndConnect();
					else if (!client) createAndConnect();
				} catch (e) { poster('reconnect failed: ' + e); }
			});
			poster('✓ reconnect handler registered');
		} catch (e) {
			poster('✗ Error registering reconnect: ' + e);
		}
		
		try {
			maxApi.addHandler('stop', function() {
				try {
					if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
					if (client && !client.destroyed) client.destroy();
					poster('Stopped client');
				} catch (e) { poster('stop failed: ' + e); }
			});
			poster('✓ stop handler registered');
		} catch (e) {
			poster('✗ Error registering stop: ' + e);
		}
		
		// Parameter change handlers - can be called from Max with messages like: temperature 1.5
		maxApi.addHandler('mode', function(val) {
			// Accept: 0 = generate, 1 = interact, or string 'generate'/'interact'
			// Python server now handles both numeric and string formats
			sendParameter({mode: val});
		});
		maxApi.addHandler('user_note', function() {
			// Accept note-ons as either:
			//   user_note pitch
			//   user_note pitch velocity
			//   user_note pitch velocity channel
			//   user_note pitch velocity duration_ms channel
			// If velocity is 0, treat it as a note-off so live durations can be captured.
			var args = Array.prototype.slice.call(arguments);
			var velocity = parseFiniteNumber(args[1]);
			if (velocity !== null && velocity <= 0) {
				sendLiveNote('user_note_off', args);
			} else {
				sendLiveNote('user_note', args);
			}
		});
		maxApi.addHandler('user_note_on', function() {
			sendLiveNote('user_note_on', Array.prototype.slice.call(arguments));
		});
		maxApi.addHandler('user_note_off', function() {
			sendLiveNote('user_note_off', Array.prototype.slice.call(arguments));
		});
		maxApi.addHandler('train_note', function(note) {
			// Send training note to server in train mode
			sendParameter({train_note: parseInt(note)});
		});
		maxApi.addHandler('train_control', function(control) {
			// Send training control command to server
			// Valid controls: start_sequence, end_sequence, reset, get_status, save_dataset
			sendParameter({train_control: String(control)});
		});
		maxApi.addHandler('temperature', function(val) {
			sendParameter({temperature: parseFloat(val)});
		});
		maxApi.addHandler('min_midi', function(val) {
			sendParameter({min_midi: parseInt(val)});
		});
		maxApi.addHandler('max_midi', function(val) {
			sendParameter({max_midi: parseInt(val)});
		});
		maxApi.addHandler('tempo', function(val) {
			sendParameter({tempo: parseFloat(val)});
		});
		maxApi.addHandler('sequencer_phase', function() {
			storeSequencerPhase(arguments);
		});
		maxApi.addHandler('note_duration_division', function(val) {
			sendParameter({note_duration_division: parseFloat(val)});
		});
		maxApi.addHandler('note_interval_division', function(val) {
			sendParameter({note_interval_division: parseFloat(val)});
		});
		maxApi.addHandler('use_probabilistic', function(val) {
			sendParameter({use_probabilistic: val ? true : false});
		});
		maxApi.addHandler('sequencer_running', function(val) {
			var normalized = String(val).toLowerCase();
			var running = !(normalized === '0' || normalized === 'false' || normalized === 'off');
			var params = {
				sequencer_running: running,
				client_time_ms: Date.now()
			};
			if (latestSequencerPhase) {
				params.sequencer_phase = latestSequencerPhase;
			}
			sendParameter(params);
		});
		maxApi.addHandler('max_history', function(val) {
			sendParameter({max_history: parseInt(val)});
		});
		maxApi.addHandler('model_index', function(val) {
			sendParameter({model_index: parseInt(val)});
		});
		maxApi.addHandler('list_models', function() {
			sendParameter({list_models: true});
		});
		maxApi.addHandler('get_current_model', function() {
			sendParameter({get_current_model: true});
		});
		
		// =============================================
		// MIDI File Training Handlers
		// =============================================
		
		maxApi.addHandler('add_training_file', function() {
			// Add a MIDI file path to the training queue
			// Accepts full path to .mid/.midi file
			var filePath = normalizePathArgs(arguments);
			poster('add_training_file called with: ' + filePath);
			sendParameter({add_training_file: filePath});
		});
		
		maxApi.addHandler('clear_training_files', function() {
			// Clear all pending training files
			poster('clear_training_files called');
			sendParameter({clear_training_files: true});
		});
		
		maxApi.addHandler('get_training_status', function() {
			// Get current training file status
			poster('get_training_status called');
			sendParameter({get_training_status: true});
		});
		
		maxApi.addHandler('set_dataset_name', function(name) {
			// Set the name for the trained model/dataset
			poster('set_dataset_name called with: ' + name);
			sendParameter({set_dataset_name: String(name)});
		});
		
		maxApi.addHandler('set_augmented', function(val) {
			// Enable/disable data augmentation during training
			poster('set_augmented called with: ' + val);
			sendParameter({set_augmented: val ? true : false});
		});
		
		maxApi.addHandler('set_viewpoint', function(val) {
			// Set viewpoint preset: 'midi', 'pitch', or 'interval'
			poster('set_viewpoint called with: ' + val);
			sendParameter({set_viewpoint: String(val)});
		});
		
		maxApi.addHandler('set_orders', function(val) {
			// Set Markov orders, e.g. "1,2,3,4,5"
			poster('set_orders called with: ' + val);
			sendParameter({set_orders: String(val)});
		});
		
		maxApi.addHandler('start_training', function() {
			// Start training from the queued MIDI files
			poster('start_training called');
			sendParameter({start_training: true});
		});
		
		maxApi.addHandler('reset', function() {
			// Reset the STM (Short-Term Memory) history of notes
			poster('reset called - clearing note history');
			sendParameter({reset: true});
		});
		
		maxApi.addHandler('load_model_by_name', function(modelName) {
			// Load a model by dataset name or "dataset/folder" path
			poster('load_model_by_name called with: ' + modelName);
			sendParameter({load_model_by_name: String(modelName)});
		});
		
		maxApi.addHandler('analyse_and_generate', function() {
			// Analyze a MIDI file note-by-note and generate IC for each note
			var filePath = normalizePathArgs(arguments);
			poster('analyse_and_generate called with: ' + filePath);
			sendParameter({analyse_and_generate: filePath});
		});
		
		maxApi.addHandler('generate_midi_file', function() {
			// Generate next note(s) and save extended MIDI file
			// Receives: path int1 int2 int3 int4 int5 int6 int7 int8 int9
			// Optional: strategies as space-separated integers for 8-note generation
			var parsed = splitPathAndStrategies(arguments);
			var outputFolder = parsed.path;
			var strategies = parsed.strategies;
			poster('generate_midi_file handler called with path: ' + outputFolder);
			poster('  extra args: ' + JSON.stringify(strategies));
			
			let params = {generate_midi_file: outputFolder};
			if (strategies && strategies.length > 0) {
				let strategiesStr = strategies.map(s => String(s)).join(' ');
				params.sampling_strategies = strategiesStr;
				poster('  sampling_strategies: ' + strategiesStr);
			}
			sendParameter(params);
		});
		
		maxApi.addHandler('variate_midi_file', function() {
			// Variate: modify notes in a specific part of the MIDI file
			// Receives: outputFolder part surpriseness
			// Example: "C:/path/to/folder" 0 37
			// part: which quarter of the file (0=beginning, 1=first_half, 2=third_quarter, 3=end)
			// surpriseness: 0-42=most probable, 43-84=average, 85-127=least probable
			var parsed = splitPathAndTrailingArgs(arguments, 2);
			var outputFolder = parsed.path;
			var part = parsed.trailing[0];
			var surpriseness = parsed.trailing[1];
			poster('variate_midi_file handler called');
			poster('  outputFolder: ' + outputFolder + ' (type: ' + typeof outputFolder + ')');
			poster('  part: ' + part + ' (type: ' + typeof part + ')');
			poster('  surpriseness: ' + surpriseness + ' (type: ' + typeof surpriseness + ')');
			
			let params = {
				variate_midi_file: outputFolder,
				part: String(part),
				surpriseness: String(surpriseness)
			};
			sendParameter(params);
			poster('  Sent variate params: folder=' + outputFolder + ', part=' + part + ', surpriseness=' + surpriseness);
		});
		
		// Also catch via 'anything' handler to ensure we get all values
		try {
			maxApi.addHandler('anything', function(selector, ...args) {
				if (selector === 'generate_midi_file') {
					poster('*** ANYTHING: generate_midi_file with selector: ' + selector);
					poster('*** ANYTHING: full args array: ' + JSON.stringify(args));
					poster('*** ANYTHING: args length: ' + args.length);
					
					// Log each individual arg
					for (let i = 0; i < args.length; i++) {
						poster('*** ANYTHING: args[' + i + '] = ' + args[i] + ' (type: ' + typeof args[i] + ')');
					}
					
					if (args.length > 0) {
						let parsed = splitPathAndStrategies(args);
						let params = {generate_midi_file: parsed.path};
						poster('*** ANYTHING: using normalized path: ' + parsed.path);
						
						// Remaining args are strategies
						if (parsed.strategies.length > 0) {
							let strategies = parsed.strategies;
							poster('*** ANYTHING: strategies array: ' + JSON.stringify(strategies));
							let strategiesStr = strategies.map(s => String(s)).join(' ');
							params.sampling_strategies = strategiesStr;
							poster('*** ANYTHING: sampling_strategies string: ' + strategiesStr + ' (' + strategies.length + ' values)');
						}
						sendParameter(params);
					}
				} else if (selector === 'variate_midi_file') {
					poster('*** ANYTHING: variate_midi_file detected');
					poster('*** ANYTHING: args: ' + JSON.stringify(args));
					poster('*** ANYTHING: args length: ' + args.length);
					
					if (args.length >= 3) {
						let parsed = splitPathAndTrailingArgs(args, 2);
						let outputFolder = parsed.path;
						let part = parseInt(parsed.trailing[0]);
						let surpriseness = parseInt(parsed.trailing[1]);
						let params = {variate_midi_file: outputFolder, part: String(part), surpriseness: String(surpriseness)};
						sendParameter(params);
						poster('*** ANYTHING: sent variate - folder=' + outputFolder + ', part=' + part + ', surpriseness=' + surpriseness);
					} else {
						poster('*** ANYTHING: variate_midi_file requires 3 args (folder, part, surpriseness), got ' + args.length);
					}
				}
			});
			poster('✓ anything handler registered for generate_midi_file fallback');
		} catch (e) {
			poster('✗ Error registering anything handler: ' + e);
		}
		
		// Handle list messages from [prepend export_path] in Max
		// Debug: catch export_path selector directly
		try {
			maxApi.addHandler('export_path', function() {
				var folderPath = normalizePathArgs(arguments);
				poster('*** EXPORT_PATH HANDLER TRIGGERED ***');
				poster('  folderPath type: ' + typeof folderPath);
				poster('  folderPath value: ' + folderPath);
				if (folderPath) {
					sendParameter({export_plot: folderPath});
				}
			});
			poster('✓ export_path handler registered');
		} catch (e) {
			poster('✗ Error registering export_path: ' + e);
		}

		// Fallback for 'anything' - debug what comes in
		try {
			maxApi.addHandler('anything', function(selector, ...rest) {
				poster('*** ANYTHING HANDLER: selector=' + selector + ', rest=' + JSON.stringify(rest));
				if (selector === 'export_path' && rest.length > 0) {
					var folderPath = normalizePathArgs(rest);
					poster('*** EXPORT via anything: ' + folderPath);
					sendParameter({export_plot: folderPath});
				}
			});
			poster('✓ anything handler registered');
		} catch (e) {
			poster('✗ Error registering anything: ' + e);
		}

		// Signal that we're ready to receive messages
		maxApi.post('listen.js is ready');
		maxApi.outlet('ready', 1);
	}
}
