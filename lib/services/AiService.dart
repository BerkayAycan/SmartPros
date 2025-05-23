import 'dart:convert';
import 'package:flutter/foundation.dart'; // debugPrint için
import 'package:http/http.dart' as http;

class AiService {
  static Future<Map<String, dynamic>?> fetchSummary({
    required String name,
    String? gender,
    String? age,
    String? weight,
  }) async {
    final queryParams = {
      'drug_name': name,
      if (gender != null && gender.isNotEmpty) 'gender': gender,
      if (age != null && age.isNotEmpty) 'age': age,
      if (weight != null && weight.isNotEmpty) 'weight': weight,
    };

    final uri = Uri.http('127.0.0.1:8000', '/summary', queryParams);

    try {
      final response = await http.get(uri);

      debugPrint("🧪 API status: ${response.statusCode}");
      debugPrint("🧪 API response: ${response.body}");

      if (response.statusCode == 200) {
        final decoded = jsonDecode(utf8.decode(response.bodyBytes));
        if (decoded.containsKey('summaries')) {
          return {
            'drugName': decoded['drugName'],
            'summaries': Map<String, String>.from(decoded['summaries']),
          };
        } else {
          debugPrint("⚠️ 'summaries' key yok!");
          return null;
        }
      } else {
        debugPrint("⚠️ API hata kodu: ${response.statusCode}");
        return null;
      }
    } catch (e) {
      debugPrint("⚠️ API çağrısı başarısız: $e");
      return null;
    }
  }
}
