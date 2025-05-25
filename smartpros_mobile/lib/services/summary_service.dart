import 'dart:convert';
import 'package:http/http.dart' as http;

class SummaryService {
  static const String baseUrl = "http://10.0.2.2:5000/api/summary";

  static Future<String> getSummary(
    Map<String, dynamic> userData,
    String drugInfo,
  ) async {
    final response = await http.post(
      Uri.parse(baseUrl),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'userData': userData, 'drugInfo': drugInfo}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['summaryText'];
    } else {
      throw Exception('Özet alınamadı');
    }
  }
}
