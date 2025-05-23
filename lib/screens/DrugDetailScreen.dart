import 'package:flutter/material.dart';

class DrugDetailScreen extends StatefulWidget {
  final String drugName;
  final List<String> purposes;
  final Map<String, String> summaries;

  const DrugDetailScreen({
    super.key,
    required this.drugName,
    required this.purposes,
    required this.summaries,
  });

  @override
  State<DrugDetailScreen> createState() => _DrugDetailScreenState();
}

class _DrugDetailScreenState extends State<DrugDetailScreen> {
  String? selectedPurpose;

  @override
  void initState() {
    super.initState();
    if (widget.purposes.isNotEmpty) {
      selectedPurpose = widget.purposes.first;
    }
  }

  @override
  Widget build(BuildContext context) {
    final summary = selectedPurpose != null
        ? widget.summaries[selectedPurpose!] ?? "İlgili özet bulunamadı."
        : "Herhangi bir kullanım amacı seçilmedi.";

    return Scaffold(
      appBar: AppBar(title: const Text("İlaç Özeti")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("İlaç İsmi: ${widget.drugName}", style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 20),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(
                labelText: "Kullanım Amacı Seçin",
                border: OutlineInputBorder(),
              ),
              value: selectedPurpose,
              items: widget.purposes.map((purpose) {
                return DropdownMenuItem<String>(
                  value: purpose,
                  child: Text(purpose),
                );
              }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedPurpose = value;
                });
              },
            ),
            const SizedBox(height: 20),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  summary,
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
