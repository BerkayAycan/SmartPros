import 'dart:io';

void startBackend() async {
  final backendPath =
      'C:/Users/fatih/Desktop/smartpros-backend'; // Flutter'a göre backend konumu
  final process = await Process.start(
    'node',
    ['server.js'],
    workingDirectory: backendPath,
    runInShell: true,
  );

  process.stdout.transform(SystemEncoding().decoder).listen(print);
  process.stderr.transform(SystemEncoding().decoder).listen(print);
}
