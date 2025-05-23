import 'package:flutter/material.dart';
import '../models/drug.dart';
import '../services/CsvLoader.dart';
import '../services/AiService.dart';
import 'DrugDetailScreen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<Drug> drugs = [];
  List<Drug> filteredDrugs = [];
  String query = '';
  String? selectedGender;
  final TextEditingController ageController = TextEditingController();
  final TextEditingController weightController = TextEditingController();

  @override
  void initState() {
    super.initState();
    CsvLoader.loadDrugs().then((list) {
      setState(() {
        drugs = list;
        filteredDrugs = list;
      });
    });
  }

  void filter(String input) {
    final normalized = input.toLowerCase().trim();
    setState(() {
      query = input;
      filteredDrugs = drugs
          .where((drug) => drug.name.toLowerCase().contains(normalized))
          .toList();
    });
  }

  Future<void> showSummary(String name) async {
    showDialog(
      context: context,
      builder: (_) => const Center(child: CircularProgressIndicator()),
      barrierDismissible: false,
    );

    final result = await AiService.fetchSummary(
      name: name,
      gender: selectedGender,
      age: ageController.text,
      weight: weightController.text,
    );

    Navigator.pop(context);

    debugPrint("🧪 Backend sonucu:\n$result");

    if (result != null &&
        result['drugName'] != null &&
        result['summaries'] != null &&
        result['summaries'] is Map &&
        (result['summaries'] as Map).isNotEmpty) {
      final summariesMap = Map<String, String>.from(result['summaries']);
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => DrugDetailScreen(
            drugName: result['drugName'],
            purposes: summariesMap.keys.toList(),
            summaries: summariesMap,
          ),
        ),
      );
    } else {
      debugPrint("⚠️ Özet verisi alınamadı. Backend dönüşü null ya da eksik.");
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("Hata"),
          content: const Text("Özet verisi alınamadı."),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text("Kapat"),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("İlaçlar")),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(10),
            child: TextField(
              onChanged: filter,
              decoration: const InputDecoration(
                labelText: 'İlaç Ara',
                border: OutlineInputBorder(),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            child: DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Cinsiyet"),
              value: selectedGender,
              items: ['male', 'female']
                  .map((g) => DropdownMenuItem(value: g, child: Text(g)))
                  .toList(),
              onChanged: (val) => setState(() => selectedGender = val),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            child: TextField(
              controller: ageController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: "Yaş"),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            child: TextField(
              controller: weightController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: "Kilo (kg)"),
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: filteredDrugs.length,
              itemBuilder: (_, index) {
                final drug = filteredDrugs[index];
                return ListTile(
                  title: Text(drug.name),
                  trailing: ElevatedButton(
                    child: const Text("Özet"),
                    onPressed: () => showSummary(drug.name),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
