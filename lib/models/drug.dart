class Drug {
  final String name;

  Drug({required this.name});

  factory Drug.fromCsv(List<dynamic> header, List<dynamic> row) {
    final nameIndex = header.indexOf('ilaç adı');
    return Drug(
      name: row[nameIndex].toString().trim().replaceAll(RegExp(r'\s+'), ' '),
    );
  }
}
