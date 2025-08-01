import 'package:immich_mobile/domain/models/log.model.dart';
import 'package:immich_mobile/infrastructure/entities/log.entity.dart';
import 'package:immich_mobile/infrastructure/repositories/db.repository.dart';
import 'package:isar/isar.dart';

class IsarLogRepository extends IsarDatabaseRepository {
  final Isar _db;
  const IsarLogRepository(super.db) : _db = db;

  Future<bool> deleteAll() async {
    await transaction(() async => await _db.loggerMessages.clear());
    return true;
  }

  Future<List<LogMessage>> getAll() async {
    final logs = await _db.loggerMessages.where().sortByCreatedAtDesc().findAll();
    return logs.map((l) => l.toDto()).toList();
  }

  Future<bool> insert(LogMessage log) async {
    final logEntity = LoggerMessage.fromDto(log);
    await transaction(() async {
      await _db.loggerMessages.put(logEntity);
    });
    return true;
  }

  Future<bool> insertAll(Iterable<LogMessage> logs) async {
    await transaction(() async {
      final logEntities = logs.map((log) => LoggerMessage.fromDto(log)).toList();
      await _db.loggerMessages.putAll(logEntities);
    });
    return true;
  }

  Future<void> truncate({int limit = 250}) async {
    await transaction(() async {
      final count = await _db.loggerMessages.count();
      if (count <= limit) return;
      final toRemove = count - limit;
      await _db.loggerMessages.where().limit(toRemove).deleteAll();
    });
  }
}
