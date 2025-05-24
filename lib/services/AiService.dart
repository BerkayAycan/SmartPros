import 'dart:convert';
import 'package:http/http.dart' as http;

class AiService {
  static Future<Map<String, dynamic>?> fetchSummary({
    required String name,
    String? gender,
    String? age,
    String? weight,
    String? allergies,
    String? conditions,
    String? currentDrugs,
    bool? isPregnant,
    bool? isDriving,
  }) async {
    final queryParams = {
      'drug_name': name,
      if (gender != null && gender.isNotEmpty)
        'gender': gender == 'erkek' ? 'male' : 'female',
      if (age != null && age.isNotEmpty) 'age': age,
      if (weight != null && weight.isNotEmpty) 'weight': weight,
      if (allergies != null && allergies.isNotEmpty) 'allergies': allergies,
      if (conditions != null && conditions.isNotEmpty) 'conditions': conditions,
      if (currentDrugs != null && currentDrugs.isNotEmpty)
        'current_drugs': currentDrugs,
      if (isPregnant != null) 'pregnant': isPregnant.toString(),
      if (isDriving != null) 'driving': isDriving.toString(),
    };

    final uri = Uri.http('127.0.0.1:8000', '/summary', queryParams);

    try {
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final decoded = jsonDecode(utf8.decode(response.bodyBytes));

        if (decoded.containsKey('summaries')) {
          return {
            'drugName': decoded['drugName'],
            'purposes': decoded['summaries'].keys.toList(),
            'summaries': Map<String, String>.from(decoded['summaries']),
          };
        } else {
          return null;
        }
      } else {
        return null;
      }
    } catch (e) {
      print("AI Service error: $e");
      return null;
    }
  }
}
