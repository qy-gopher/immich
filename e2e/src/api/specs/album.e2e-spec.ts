import {
  addAssetsToAlbum,
  AlbumResponseDto,
  AlbumUserRole,
  AssetMediaResponseDto,
  AssetOrder,
  deleteUserAdmin,
  getAlbumInfo,
  LoginResponseDto,
  SharedLinkType,
} from '@immich/sdk';
import { createUserDto } from 'src/fixtures';
import { errorDto } from 'src/responses';
import { app, asBearerAuth, utils } from 'src/utils';
import request from 'supertest';
import { beforeAll, beforeEach, describe, expect, it } from 'vitest';

const user1SharedEditorUser = 'user1SharedEditorUser';
const user1SharedViewerUser = 'user1SharedViewerUser';
const user1SharedLink = 'user1SharedLink';
const user1NotShared = 'user1NotShared';
const user2SharedUser = 'user2SharedUser';
const user2SharedLink = 'user2SharedLink';
const user2NotShared = 'user2NotShared';
const user4DeletedAsset = 'user4DeletedAsset';
const user4Empty = 'user4Empty';

describe('/albums', () => {
  let admin: LoginResponseDto;
  let user1: LoginResponseDto;
  let user1Asset1: AssetMediaResponseDto;
  let user1Asset2: AssetMediaResponseDto;
  let user4Asset1: AssetMediaResponseDto;
  let user1Albums: AlbumResponseDto[];
  let user2: LoginResponseDto;
  let user2Albums: AlbumResponseDto[];
  let deletedAssetAlbum: AlbumResponseDto;
  let user3: LoginResponseDto; // deleted
  let user4: LoginResponseDto;

  beforeAll(async () => {
    await utils.resetDatabase();

    admin = await utils.adminSetup();

    [user1, user2, user3, user4] = await Promise.all([
      utils.userSetup(admin.accessToken, createUserDto.user1),
      utils.userSetup(admin.accessToken, createUserDto.user2),
      utils.userSetup(admin.accessToken, createUserDto.user3),
      utils.userSetup(admin.accessToken, createUserDto.user4),
    ]);

    [user1Asset1, user1Asset2, user4Asset1] = await Promise.all([
      utils.createAsset(user1.accessToken, { isFavorite: true }),
      utils.createAsset(user1.accessToken),
      utils.createAsset(user1.accessToken),
    ]);

    [user1Albums, user2Albums, deletedAssetAlbum] = await Promise.all([
      Promise.all([
        utils.createAlbum(user1.accessToken, {
          albumName: user1SharedEditorUser,
          albumUsers: [
            { userId: admin.userId, role: AlbumUserRole.Editor },
            { userId: user2.userId, role: AlbumUserRole.Editor },
          ],
          assetIds: [user1Asset1.id],
        }),
        utils.createAlbum(user1.accessToken, {
          albumName: user1SharedLink,
          assetIds: [user1Asset1.id],
        }),
        utils.createAlbum(user1.accessToken, {
          albumName: user1NotShared,
          assetIds: [user1Asset1.id, user1Asset2.id],
        }),
        utils.createAlbum(user1.accessToken, {
          albumName: user1SharedViewerUser,
          albumUsers: [{ userId: user2.userId, role: AlbumUserRole.Viewer }],
          assetIds: [user1Asset1.id],
        }),
      ]),
      Promise.all([
        utils.createAlbum(user2.accessToken, {
          albumName: user2SharedUser,
          albumUsers: [
            { userId: user1.userId, role: AlbumUserRole.Editor },
            { userId: user3.userId, role: AlbumUserRole.Editor },
          ],
        }),
        utils.createAlbum(user2.accessToken, { albumName: user2SharedLink }),
        utils.createAlbum(user2.accessToken, { albumName: user2NotShared }),
      ]),
      utils.createAlbum(user4.accessToken, { albumName: user4DeletedAsset }),
      utils.createAlbum(user4.accessToken, { albumName: user4Empty }),
      utils.createAlbum(user3.accessToken, {
        albumName: 'Deleted',
        albumUsers: [{ userId: user1.userId, role: AlbumUserRole.Editor }],
      }),
    ]);

    await Promise.all([
      addAssetsToAlbum(
        { id: user2Albums[0].id, bulkIdsDto: { ids: [user1Asset1.id, user1Asset2.id] } },
        { headers: asBearerAuth(user1.accessToken) },
      ),
      addAssetsToAlbum(
        { id: deletedAssetAlbum.id, bulkIdsDto: { ids: [user4Asset1.id] } },
        { headers: asBearerAuth(user4.accessToken) },
      ),
      // add shared link to user1SharedLink album
      utils.createSharedLink(user1.accessToken, {
        type: SharedLinkType.Album,
        albumId: user1Albums[1].id,
      }),
      // add shared link to user2SharedLink album
      utils.createSharedLink(user2.accessToken, {
        type: SharedLinkType.Album,
        albumId: user2Albums[1].id,
      }),
    ]);

    [user2Albums[0]] = await Promise.all([
      getAlbumInfo({ id: user2Albums[0].id }, { headers: asBearerAuth(user2.accessToken) }),
      deleteUserAdmin({ id: user3.userId, userAdminDeleteDto: {} }, { headers: asBearerAuth(admin.accessToken) }),
      utils.deleteAssets(user1.accessToken, [user4Asset1.id]),
    ]);
  });

  describe('GET /albums', () => {
    it("should not show other users' favorites", async () => {
      const { status, body } = await request(app)
        .get(`/albums/${user1Albums[0].id}?withoutAssets=false`)
        .set('Authorization', `Bearer ${user2.accessToken}`);
      expect(status).toEqual(200);
      expect(body).toEqual({
        ...user1Albums[0],
        assets: [expect.objectContaining({ isFavorite: false })],
        lastModifiedAssetTimestamp: expect.any(String),
        startDate: expect.any(String),
        endDate: expect.any(String),
        shared: true,
        albumUsers: expect.any(Array),
      });
    });

    it('should not return shared albums with a deleted owner', async () => {
      const { status, body } = await request(app)
        .get('/albums?shared=true')
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toHaveLength(4);
      expect(body).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedLink,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedEditorUser,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedViewerUser,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user2.userId,
            albumName: user2SharedUser,
            shared: true,
          }),
        ]),
      );
    });

    it('should return the album collection including owned and shared', async () => {
      const { status, body } = await request(app).get('/albums').set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(200);
      expect(body).toHaveLength(4);
      expect(body).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedEditorUser,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedViewerUser,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedLink,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1NotShared,
            shared: false,
          }),
        ]),
      );
    });

    it('should return the album collection filtered by shared', async () => {
      const { status, body } = await request(app)
        .get('/albums?shared=true')
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(200);
      expect(body).toHaveLength(4);
      expect(body).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedEditorUser,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedViewerUser,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1SharedLink,
            shared: true,
          }),
          expect.objectContaining({
            ownerId: user2.userId,
            albumName: user2SharedUser,
            shared: true,
          }),
        ]),
      );
    });

    it('should return the album collection filtered by NOT shared', async () => {
      const { status, body } = await request(app)
        .get('/albums?shared=false')
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(200);
      expect(body).toHaveLength(1);
      expect(body).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            ownerId: user1.userId,
            albumName: user1NotShared,
            shared: false,
          }),
        ]),
      );
    });

    it('should return the album collection filtered by assetId', async () => {
      const { status, body } = await request(app)
        .get(`/albums?assetId=${user1Asset2.id}`)
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(200);
      expect(body).toHaveLength(2);
    });

    it('should return the album collection filtered by assetId and ignores shared=true', async () => {
      const { status, body } = await request(app)
        .get(`/albums?shared=true&assetId=${user1Asset1.id}`)
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(200);
      expect(body).toHaveLength(5);
    });

    it('should return the album collection filtered by assetId and ignores shared=false', async () => {
      const { status, body } = await request(app)
        .get(`/albums?shared=false&assetId=${user1Asset1.id}`)
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(200);
      expect(body).toHaveLength(5);
    });

    it('should return empty albums and albums where all assets are deleted', async () => {
      const { status, body } = await request(app).get('/albums').set('Authorization', `Bearer ${user4.accessToken}`);
      expect(status).toBe(200);
      expect(body).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            ownerId: user4.userId,
            albumName: user4DeletedAsset,
            shared: false,
          }),
          expect.objectContaining({
            ownerId: user4.userId,
            albumName: user4Empty,
            shared: false,
          }),
        ]),
      );
    });
  });

  describe('GET /albums/:id', () => {
    it('should return album info for own album', async () => {
      const { status, body } = await request(app)
        .get(`/albums/${user1Albums[0].id}?withoutAssets=false`)
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toEqual({
        ...user1Albums[0],
        assets: [expect.objectContaining({ id: user1Albums[0].assets[0].id })],
        lastModifiedAssetTimestamp: expect.any(String),
        startDate: expect.any(String),
        endDate: expect.any(String),
        albumUsers: expect.any(Array),
        shared: true,
      });
    });

    it('should return album info for shared album (editor)', async () => {
      const { status, body } = await request(app)
        .get(`/albums/${user2Albums[0].id}?withoutAssets=false`)
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toMatchObject({ id: user2Albums[0].id });
    });

    it('should return album info for shared album (viewer)', async () => {
      const { status, body } = await request(app)
        .get(`/albums/${user1Albums[3].id}?withoutAssets=false`)
        .set('Authorization', `Bearer ${user2.accessToken}`);

      expect(status).toBe(200);
      expect(body).toMatchObject({ id: user1Albums[3].id });
    });

    it('should return album info with assets when withoutAssets is undefined', async () => {
      const { status, body } = await request(app)
        .get(`/albums/${user1Albums[0].id}`)
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toEqual({
        ...user1Albums[0],
        assets: [expect.objectContaining({ id: user1Albums[0].assets[0].id })],
        lastModifiedAssetTimestamp: expect.any(String),
        startDate: expect.any(String),
        endDate: expect.any(String),
        albumUsers: expect.any(Array),
        shared: true,
      });
    });

    it('should return album info without assets when withoutAssets is true', async () => {
      const { status, body } = await request(app)
        .get(`/albums/${user1Albums[0].id}?withoutAssets=true`)
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toEqual({
        ...user1Albums[0],
        assets: [],
        assetCount: 1,
        lastModifiedAssetTimestamp: expect.any(String),
        endDate: expect.any(String),
        startDate: expect.any(String),
        albumUsers: expect.any(Array),
        shared: true,
      });
    });

    it('should not count trashed assets', async () => {
      await utils.deleteAssets(user1.accessToken, [user1Asset2.id]);

      const { status, body } = await request(app)
        .get(`/albums/${user2Albums[0].id}?withoutAssets=true`)
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toEqual({
        ...user2Albums[0],
        assets: [],
        assetCount: 1,
        lastModifiedAssetTimestamp: expect.any(String),
        endDate: expect.any(String),
        startDate: expect.any(String),
        albumUsers: expect.any(Array),
        shared: true,
      });
    });
  });

  describe('GET /albums/statistics', () => {
    it('should return total count of albums the user has access to', async () => {
      const { status, body } = await request(app)
        .get('/albums/statistics')
        .set('Authorization', `Bearer ${user1.accessToken}`);

      expect(status).toBe(200);
      expect(body).toEqual({ owned: 4, shared: 4, notShared: 1 });
    });
  });

  describe('POST /albums', () => {
    it('should create an album', async () => {
      const { status, body } = await request(app)
        .post('/albums')
        .send({ albumName: 'New album' })
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(201);
      expect(body).toEqual({
        id: expect.any(String),
        createdAt: expect.any(String),
        updatedAt: expect.any(String),
        ownerId: user1.userId,
        albumName: 'New album',
        description: '',
        albumThumbnailAssetId: null,
        shared: false,
        albumUsers: [],
        hasSharedLink: false,
        assets: [],
        assetCount: 0,
        owner: expect.objectContaining({ email: user1.userEmail }),
        isActivityEnabled: true,
        order: AssetOrder.Desc,
      });
    });

    it('should not be able to share album with owner', async () => {
      const { status, body } = await request(app)
        .post('/albums')
        .send({ albumName: 'New album', albumUsers: [{ role: AlbumUserRole.Editor, userId: user1.userId }] })
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Cannot share album with owner'));
    });
  });

  describe('PUT /albums/:id/assets', () => {
    it('should be able to add own asset to own album', async () => {
      const asset = await utils.createAsset(user1.accessToken);
      const { status, body } = await request(app)
        .put(`/albums/${user1Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ ids: [asset.id] });

      expect(status).toBe(200);
      expect(body).toEqual([expect.objectContaining({ id: asset.id, success: true })]);
    });

    it('should be able to add own asset to shared album', async () => {
      const asset = await utils.createAsset(user1.accessToken);
      const { status, body } = await request(app)
        .put(`/albums/${user2Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ ids: [asset.id] });

      expect(status).toBe(200);
      expect(body).toEqual([expect.objectContaining({ id: asset.id, success: true })]);
    });

    it('should not be able to add assets to album as a viewer', async () => {
      const asset = await utils.createAsset(user2.accessToken);
      const { status, body } = await request(app)
        .put(`/albums/${user1Albums[3].id}/assets`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ ids: [asset.id] });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Not found or no albumAsset.create access'));
    });

    it('should add duplicate assets only once', async () => {
      const asset = await utils.createAsset(user1.accessToken);
      const { status, body } = await request(app)
        .put(`/albums/${user1Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ ids: [asset.id, asset.id] });

      expect(status).toBe(200);
      expect(body).toEqual([
        expect.objectContaining({ id: asset.id, success: true }),
        expect.objectContaining({ id: asset.id, success: false, error: 'duplicate' }),
      ]);
    });
  });

  describe('PATCH /albums/:id', () => {
    it('should update an album', async () => {
      const album = await utils.createAlbum(user1.accessToken, {
        albumName: 'New album',
      });
      const { status, body } = await request(app)
        .patch(`/albums/${album.id}`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({
          albumName: 'New album name',
          description: 'An album description',
        });
      expect(status).toBe(200);
      expect(body).toEqual({
        ...album,
        updatedAt: expect.any(String),
        albumName: 'New album name',
        description: 'An album description',
      });
    });

    it('should not be able to update as a viewer', async () => {
      const { status, body } = await request(app)
        .patch(`/albums/${user1Albums[3].id}`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ albumName: 'New album name' });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Not found or no album.update access'));
    });

    it('should not be able to update as an editor', async () => {
      const { status, body } = await request(app)
        .patch(`/albums/${user1Albums[0].id}`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ albumName: 'New album name' });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Not found or no album.update access'));
    });
  });

  describe('DELETE /albums/:id/assets', () => {
    it('should require authorization', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user1Albums[1].id}/assets`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ ids: [user1Asset1.id] });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.noPermission);
    });

    it('should be able to remove foreign asset from owned album', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user2Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ ids: [user1Asset1.id] });

      expect(status).toBe(200);
      expect(body).toEqual([
        expect.objectContaining({
          id: user1Asset1.id,
          success: true,
        }),
      ]);
    });

    it('should not be able to remove foreign asset from foreign album', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user1Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ ids: [user1Asset1.id] });

      expect(status).toBe(200);
      expect(body).toEqual([
        expect.objectContaining({
          id: user1Asset1.id,
          success: false,
          error: 'no_permission',
        }),
      ]);
    });

    it('should be able to remove own asset from own album', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user1Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ ids: [user1Asset1.id] });

      expect(status).toBe(200);
      expect(body).toEqual([expect.objectContaining({ id: user1Asset1.id, success: true })]);
    });

    it('should be able to remove own asset from shared album', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user2Albums[0].id}/assets`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ ids: [user1Asset2.id] });

      expect(status).toBe(200);
      expect(body).toEqual([expect.objectContaining({ id: user1Asset2.id, success: true })]);
    });

    it('should not be able to remove assets from album as a viewer', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user1Albums[3].id}/assets`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ ids: [user1Asset1.id] });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Not found or no albumAsset.delete access'));
    });

    it('should remove duplicate assets only once', async () => {
      const { status, body } = await request(app)
        .delete(`/albums/${user1Albums[1].id}/assets`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ ids: [user1Asset1.id, user1Asset1.id] });

      expect(status).toBe(200);
      expect(body).toEqual([
        expect.objectContaining({ id: user1Asset1.id, success: true }),
        expect.objectContaining({ id: user1Asset1.id, success: false, error: 'not_found' }),
      ]);
    });
  });

  describe('PUT :id/users', () => {
    let album: AlbumResponseDto;

    beforeEach(async () => {
      album = await utils.createAlbum(user1.accessToken, {
        albumName: 'testAlbum',
      });
    });

    it('should be able to add user to own album', async () => {
      const { status, body } = await request(app)
        .put(`/albums/${album.id}/users`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ albumUsers: [{ userId: user2.userId, role: AlbumUserRole.Editor }] });

      expect(status).toBe(200);
      expect(body).toEqual(
        expect.objectContaining({
          albumUsers: [
            expect.objectContaining({
              user: expect.objectContaining({ id: user2.userId }),
            }),
          ],
        }),
      );
    });

    it('should not be able to share album with owner', async () => {
      const { status, body } = await request(app)
        .put(`/albums/${album.id}/users`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ albumUsers: [{ userId: user1.userId, role: AlbumUserRole.Editor }] });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Cannot be shared with owner'));
    });

    it('should not be able to add existing user to shared album', async () => {
      await request(app)
        .put(`/albums/${album.id}/users`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ albumUsers: [{ userId: user2.userId, role: AlbumUserRole.Editor }] });

      const { status, body } = await request(app)
        .put(`/albums/${album.id}/users`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ albumUsers: [{ userId: user2.userId, role: AlbumUserRole.Editor }] });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('User already added'));
    });
  });

  describe('PUT :id/user/:userId', () => {
    it('should allow the album owner to change the role of a shared user', async () => {
      const album = await utils.createAlbum(user1.accessToken, {
        albumName: 'testAlbum',
        albumUsers: [{ userId: user2.userId, role: AlbumUserRole.Viewer }],
      });

      expect(album.albumUsers[0].role).toEqual(AlbumUserRole.Viewer);

      const { status } = await request(app)
        .put(`/albums/${album.id}/user/${user2.userId}`)
        .set('Authorization', `Bearer ${user1.accessToken}`)
        .send({ role: AlbumUserRole.Editor });

      expect(status).toBe(200);

      // Get album to verify the role change
      const { body } = await request(app)
        .get(`/albums/${album.id}`)
        .set('Authorization', `Bearer ${user1.accessToken}`);
      expect(body).toEqual(
        expect.objectContaining({
          albumUsers: [expect.objectContaining({ role: AlbumUserRole.Editor })],
        }),
      );
    });

    it('should not allow a shared user to change the role of another shared user', async () => {
      const album = await utils.createAlbum(user1.accessToken, {
        albumName: 'testAlbum',
        albumUsers: [{ userId: user2.userId, role: AlbumUserRole.Viewer }],
      });

      expect(album.albumUsers[0].role).toEqual(AlbumUserRole.Viewer);

      const { status, body } = await request(app)
        .put(`/albums/${album.id}/user/${user2.userId}`)
        .set('Authorization', `Bearer ${user2.accessToken}`)
        .send({ role: AlbumUserRole.Editor });

      expect(status).toBe(400);
      expect(body).toEqual(errorDto.badRequest('Not found or no album.share access'));
    });
  });
});
