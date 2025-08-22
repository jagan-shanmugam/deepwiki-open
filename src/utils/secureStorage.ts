// Best-effort encrypted storage for sensitive client-side values (e.g., PATs)
// Notes:
// - Uses Web Crypto API (AES-GCM) with a per-origin symmetric key persisted in localStorage.
// - This improves at-rest secrecy in localStorage, but is not equivalent to a user password vault.
// - Do not rely on this for high-assurance secrets beyond the browser context.

const STORAGE_KEY_PREFIX = 'dw_pat_';
const KEY_STORAGE_NAME = 'dw_pat_key_jwk';

function isWebCryptoAvailable(): boolean {
    return typeof window !== 'undefined' && !!window.crypto && !!window.crypto.subtle;
}

async function getOrCreateCryptoKey(): Promise<CryptoKey | null> {
    if (!isWebCryptoAvailable()) return null;

    try {
        const existing = localStorage.getItem(KEY_STORAGE_NAME);
        if (existing) {
            const jwk = JSON.parse(existing);
            return await window.crypto.subtle.importKey(
                'jwk',
                jwk,
                { name: 'AES-GCM' },
                true,
                ['encrypt', 'decrypt']
            );
        }

        const key = await window.crypto.subtle.generateKey(
            { name: 'AES-GCM', length: 256 },
            true,
            ['encrypt', 'decrypt']
        );
        const jwk = await window.crypto.subtle.exportKey('jwk', key);
        localStorage.setItem(KEY_STORAGE_NAME, JSON.stringify(jwk));
        return key;
    } catch {
        return null;
    }
}

function toBase64(bytes: ArrayBuffer): string {
    const binary = String.fromCharCode(...new Uint8Array(bytes));
    return btoa(binary);
}

function fromBase64(b64: string): Uint8Array {
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

export async function saveToken(platform: 'github' | 'gitlab' | 'bitbucket', token: string): Promise<void> {
    const storageKey = `${STORAGE_KEY_PREFIX}${platform}`;

    if (!token) {
        localStorage.removeItem(storageKey);
        return;
    }

    const key = await getOrCreateCryptoKey();
    if (!key) {
        // Fallback: store as-is if crypto unavailable (last resort)
        localStorage.setItem(storageKey, token);
        return;
    }

    const iv = window.crypto.getRandomValues(new Uint8Array(12));
    const encoded = new TextEncoder().encode(token);
    const ciphertext = await window.crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encoded);

    const payload = {
        iv: toBase64(iv.buffer),
        data: toBase64(ciphertext)
    };
    localStorage.setItem(storageKey, JSON.stringify(payload));
}

export async function loadToken(platform: 'github' | 'gitlab' | 'bitbucket'): Promise<string> {
    const storageKey = `${STORAGE_KEY_PREFIX}${platform}`;
    const raw = localStorage.getItem(storageKey);
    if (!raw) return '';

    const key = await getOrCreateCryptoKey();
    if (!key) {
        // Fallback: return raw if crypto unavailable
        try { return raw.startsWith('{') ? '' : raw; } catch { return ''; }
    }

    try {
        const payload = JSON.parse(raw);
        const iv = fromBase64(payload.iv);
        const data = fromBase64(payload.data);
        const plaintext = await window.crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, data);
        return new TextDecoder().decode(plaintext);
    } catch {
        // If parsing/decryption fails, clear corrupted entry
        localStorage.removeItem(storageKey);
        return '';
    }
}

export function clearToken(platform: 'github' | 'gitlab' | 'bitbucket'): void {
    localStorage.removeItem(`${STORAGE_KEY_PREFIX}${platform}`);
}


