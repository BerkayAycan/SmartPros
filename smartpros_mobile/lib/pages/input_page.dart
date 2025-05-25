import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../services/summary_service.dart';
import 'summary_page.dart';

class InputPage extends StatefulWidget {
  @override
  _InputPageState createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _ageController = TextEditingController();
  final TextEditingController _allergiesController = TextEditingController();
  final TextEditingController _diseasesController = TextEditingController();
  final TextEditingController _currentMedicationsController =
      TextEditingController();
  final TextEditingController _drugController = TextEditingController();

  String _gender = "Kadın";
  bool _pregnancyStatus = false;
  bool _drivingStatus = false;
  bool _isLoading = false;
  List<String> _drugSuggestions = [];

  void _fetchDrugSuggestions(String input) async {
    if (input.length < 2) {
      setState(() => _drugSuggestions = []);
      return;
    }
    try {
      final response = await http
          .get(Uri.parse("http://10.0.2.2:5000/api/drugs?query=$input"));
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        setState(() {
          _drugSuggestions = data.cast<String>();
        });
      }
    } catch (e) {
      print("Hata: $e");
    }
  }

  void _submitForm() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final userData = {
        "userId": "user123",
        "age": int.parse(_ageController.text),
        "gender": _gender,
        "allergies":
            _allergiesController.text.split(',').map((e) => e.trim()).toList(),
        "diseases":
            _diseasesController.text.split(',').map((e) => e.trim()).toList(),
        "pregnancyStatus": _pregnancyStatus,
        "drivingStatus": _drivingStatus,
        "currentMedications": _currentMedicationsController.text
            .split(',')
            .map((e) => e.trim())
            .toList(),
      };

      final drugInfo = _drugController.text.trim();
      final summary = await SummaryService.getSummary(userData, drugInfo);

      Navigator.push(
        context,
        MaterialPageRoute(
            builder: (context) => SummaryPage(summaryText: summary)),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Bir hata oluştu: \${e.toString()}')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('SmartPros - Form')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              TextFormField(
                controller: _ageController,
                decoration: InputDecoration(labelText: 'Yaş'),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    value == null || value.isEmpty ? 'Yaş girin' : null,
              ),
              DropdownButtonFormField<String>(
                value: _gender,
                decoration: InputDecoration(labelText: 'Cinsiyet'),
                items: ["Kadın", "Erkek"].map((gender) {
                  return DropdownMenuItem(value: gender, child: Text(gender));
                }).toList(),
                onChanged: (value) {
                  setState(() {
                    _gender = value!;
                    if (_gender == "Erkek") {
                      _pregnancyStatus = false;
                    }
                  });
                },
              ),
              TextFormField(
                controller: _allergiesController,
                decoration: InputDecoration(labelText: 'Alerjiler (virgülle)'),
              ),
              TextFormField(
                controller: _diseasesController,
                decoration:
                    InputDecoration(labelText: 'Hastalıklar (virgülle)'),
              ),
              SwitchListTile(
                title: Text('Hamile misiniz?'),
                value: _pregnancyStatus,
                onChanged: _gender == "Erkek"
                    ? null
                    : (val) => setState(() => _pregnancyStatus = val),
              ),
              SwitchListTile(
                title: Text('Araç kullanıyor musunuz?'),
                value: _drivingStatus,
                onChanged: (val) => setState(() => _drivingStatus = val),
              ),
              TextFormField(
                controller: _currentMedicationsController,
                decoration: InputDecoration(
                    labelText: 'Kullandığınız ilaçlar (virgülle)'),
              ),
              TextFormField(
                controller: _drugController,
                onChanged: _fetchDrugSuggestions,
                decoration: InputDecoration(labelText: 'İlaç Adı'),
              ),
              ..._drugSuggestions.map((name) => ListTile(
                    title: Text(name),
                    onTap: () {
                      setState(() {
                        _drugController.text = name;
                        _drugSuggestions.clear();
                      });
                    },
                  )),
              SizedBox(height: 20),
              _isLoading
                  ? Center(child: CircularProgressIndicator())
                  : ElevatedButton(
                      onPressed: _submitForm,
                      child: Text('Özetle'),
                    ),
            ],
          ),
        ),
      ),
    );
  }
}
