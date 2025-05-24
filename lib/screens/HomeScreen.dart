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
  final TextEditingController allergyController = TextEditingController();
  final TextEditingController conditionController = TextEditingController();
  final TextEditingController drugController = TextEditingController();

  bool isPregnant = false;
  bool isDriving = false;

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
      allergies: allergyController.text,
      conditions: conditionController.text,
      currentDrugs: drugController.text,
      isPregnant: isPregnant,
      isDriving: isDriving,
    );

    Navigator.pop(context);

    if (result != null &&
        result['drugName'] != null &&
        result['summaries'] != null &&
        result['summaries'] is Map<String, dynamic>) {
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
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: ListView(
          children: [
            TextField(
              onChanged: filter,
              decoration: const InputDecoration(
                labelText: 'İlaç Ara',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 10),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Cinsiyet"),
              value: selectedGender,
              items: ['erkek', 'kadın']
                  .map((g) => DropdownMenuItem(value: g, child: Text(g)))
                  .toList(),
              onChanged: (val) {
                setState(() {
                  selectedGender = val;
                  if (val == 'erkek')
                    isPregnant = false; // erkekse otomatik kapalı
                });
              },
            ),
            TextField(
              controller: ageController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: "Yaş"),
            ),
            TextField(
              controller: weightController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(labelText: "Kilo (kg)"),
            ),
            TextField(
              controller: allergyController,
              decoration:
                  const InputDecoration(labelText: "Alerjiler (virgülle)"),
            ),
            TextField(
              controller: conditionController,
              decoration:
                  const InputDecoration(labelText: "Hastalıklar (virgülle)"),
            ),
            SwitchListTile(
              title: const Text("Hamile misiniz?"),
              value: isPregnant,
              onChanged: selectedGender == 'kadın'
                  ? (val) => setState(() => isPregnant = val)
                  : null, // disable switch if not kadın
            ),
            SwitchListTile(
              title: const Text("Araç kullanıyor musunuz?"),
              value: isDriving,
              onChanged: (val) => setState(() => isDriving = val),
            ),
            TextField(
              controller: drugController,
              decoration: const InputDecoration(
                  labelText: "Kullandığınız ilaçlar (virgülle)"),
            ),
            const SizedBox(height: 15),
            const Divider(),
            const Text("İlaç Listesi",
                style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            ...filteredDrugs.map((drug) {
              return ListTile(
                title: Text(drug.name),
                trailing: ElevatedButton(
                  child: const Text("Özet"),
                  onPressed: () => showSummary(drug.name),
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }
}
