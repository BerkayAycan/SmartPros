import 'dart:convert';
import 'package:csv/csv.dart';
import 'package:flutter/services.dart';
import '../models/drug.dart';

class CsvLoader {
  static Future<List<Drug>> loadDrugs() async {
    final rawData = await rootBundle.loadString('assets/DrugsData_cleaned.csv');
    final List<List<dynamic>> csvTable = CsvToListConverter(eol: '\n').convert(rawData);

    final header = csvTable[0];
    final rows = csvTable.sublist(1);

    return rows.map((row) => Drug.fromCsv(header, row)).toList();
  }
}
