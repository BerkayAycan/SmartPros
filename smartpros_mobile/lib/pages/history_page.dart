import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'summary_page.dart';

class HistoryPage extends StatefulWidget {
  final String userId;
  const HistoryPage({required this.userId});

  @override
  _HistoryPageState createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  List<dynamic> _history = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  void _fetchHistory() async {
    final response = await http
        .get(Uri.parse("http://10.0.2.2:5000/api/history/${widget.userId}"));

    if (response.statusCode == 200) {
      setState(() {
        _history = jsonDecode(response.body);
        _isLoading = false;
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Geçmiş alınamadı.")),
      );
      setState(() => _isLoading = false);
    }
  }

  void _openSummary(String summaryText) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => SummaryPage(summaryText: summaryText),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Özet Geçmişi")),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : _history.isEmpty
              ? Center(child: Text("Henüz özet yok."))
              : ListView.builder(
                  itemCount: _history.length,
                  itemBuilder: (context, index) {
                    final item = _history[index];
                    return ListTile(
                      title: Text(item['drugName'] ?? 'İlaç'),
                      subtitle: Text(item['createdAt'].substring(0, 10)),
                      onTap: () => _openSummary(item['summaryText']),
                    );
                  },
                ),
    );
  }
}
