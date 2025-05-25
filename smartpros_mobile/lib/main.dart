import 'package:flutter/material.dart';
import 'pages/input_page.dart';
import 'backend_runner.dart';

void main() {
  startBackend(); // ✅ Backend otomatik başlasın
  runApp(SmartProsApp());
}

class SmartProsApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartPros',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: InputPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}
