import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(const PlantDoctorApp());

class Prediction {
  final String label;
  final double confidence;
  Prediction(this.label, this.confidence);
}

class Result {
  final List<Prediction> top;
  final int latencyMs;
  final bool uncertain;
  Result(this.top, this.latencyMs, this.uncertain);
}

/// Fully offline classifier: squash-resize to 224x224, float 0-255 input (model normalises internally),
/// temperature-scaled confidence, and an "unsure" fallback below [threshold].
class Classifier {
  static const int size = 224;
  late Interpreter _interp;
  late List<String> labels;
  double temperature = 1.0;
  double threshold = 0.6;

  Future<void> load() async {
    _interp = await Interpreter.fromAsset('assets/model.tflite',
        options: InterpreterOptions()..threads = 2);
    labels = (await rootBundle.loadString('assets/labels.txt'))
        .split('\n')
        .where((s) => s.trim().isNotEmpty)
        .toList();
    final cal = jsonDecode(await rootBundle.loadString('assets/calibration.json'));
    temperature = (cal['temperature'] as num).toDouble();
    threshold = (cal['threshold'] as num).toDouble();
  }

  Result classify(File file) {
    final decoded = img.decodeImage(file.readAsBytesSync())!;
    final resized = img.copyResize(img.bakeOrientation(decoded), width: size, height: size);
    final input = [
      List.generate(size, (y) => List.generate(size, (x) {
            final p = resized.getPixel(x, y);
            return [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
          }))
    ];
    final output = [List<double>.filled(labels.length, 0.0)];
    final sw = Stopwatch()..start();
    _interp.run(input, output);
    sw.stop();

    final logits = output[0].map((p) => math.log(math.max(p, 1e-8)) / temperature).toList();
    final m = logits.reduce(math.max);
    final exps = logits.map((l) => math.exp(l - m)).toList();
    final sum = exps.reduce((a, b) => a + b);
    final preds = List.generate(labels.length, (i) => Prediction(labels[i], exps[i] / sum))
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    return Result(preds.take(3).toList(), sw.elapsedMilliseconds, preds.first.confidence < threshold);
  }
}

class PlantDoctorApp extends StatelessWidget {
  const PlantDoctorApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Plant Doctor',
        theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
        home: const HomePage(),
      );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _clf = Classifier();
  final _picker = ImagePicker();
  bool _ready = false;
  File? _image;
  Result? _result;
  String? _error;

  @override
  void initState() {
    super.initState();
    _clf.load().then((_) => setState(() => _ready = true)).catchError((e) => setState(() => _error = '$e'));
  }

  Future<void> _pick(ImageSource src) async {
    final x = await _picker.pickImage(source: src, maxWidth: 1600);
    if (x == null) return;
    final f = File(x.path);
    setState(() {
      _image = f;
      _result = null;
    });
    try {
      final r = _clf.classify(f);
      setState(() => _result = r);
    } catch (e) {
      setState(() => _error = '$e');
    }
  }

  String _pretty(String s) => s.replaceAll('_', ' ');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Plant Doctor (offline)')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
        AspectRatio(
          aspectRatio: 1,
          child: Container(
            decoration: BoxDecoration(color: Colors.black12, borderRadius: BorderRadius.circular(12)),
            clipBehavior: Clip.antiAlias,
            child: _image == null
                ? const Center(child: Text('Take or choose a photo of a single leaf'))
                : Image.file(_image!, fit: BoxFit.cover),
          ),
        ),
        const SizedBox(height: 12),
        Row(children: [
          Expanded(
              child: FilledButton.icon(
                  onPressed: _ready ? () => _pick(ImageSource.camera) : null,
                  icon: const Icon(Icons.photo_camera),
                  label: const Text('Camera'))),
          const SizedBox(width: 12),
          Expanded(
              child: OutlinedButton.icon(
                  onPressed: _ready ? () => _pick(ImageSource.gallery) : null,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'))),
        ]),
        const SizedBox(height: 16),
        if (_result != null) ...[
          if (_result!.uncertain)
            const Card(
                color: Color(0xFFFFF3CD),
                child: Padding(
                    padding: EdgeInsets.all(12),
                    child: Text('Not sure. Retake the photo: one leaf, good light, fill the frame.'))),
          for (final p in _result!.top)
            ListTile(
              title: Text(_pretty(p.label)),
              subtitle: LinearProgressIndicator(value: p.confidence),
              trailing: Text('${(p.confidence * 100).toStringAsFixed(0)}%'),
            ),
          Text('Inference: ${_result!.latencyMs} ms (on device, no network)',
              style: Theme.of(context).textTheme.bodySmall),
        ],
        const SizedBox(height: 16),
        Text('Screening aid only. It is not a substitute for an agronomist.',
            style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}
